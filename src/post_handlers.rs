use chrono::{Datelike, Utc};
use rand::{distributions::Alphanumeric, Rng};
use rouille::{Request, Response};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    fs::{self, OpenOptions},
    io::Read,
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex},
};

const MAX_UPLOAD_SIZE_BYTES: usize = 10 * 1024 * 1024;
const MAX_NOTE_SIZE_BYTES: usize = 64 * 1024;
const ALLOWED_IMAGE_MIME_TYPES: [&str; 4] = ["image/png", "image/jpeg", "image/webp", "image/gif"];
const ALLOWED_AUDIO_MIME_TYPES: [&str; 4] = ["audio/mpeg", "audio/wav", "audio/ogg", "audio/webm"];

#[derive(Deserialize)]
struct JsonPostBody {
    text: String,
}

#[derive(Serialize)]
struct ApiResponse {
    success: bool,
    message: String,
}

pub fn handle_post_request(req: &Request, tags_arc: &Arc<Mutex<HashSet<String>>>) -> Response {
    let content_type = req
        .header("Content-Type")
        .map(|v| v.to_ascii_lowercase())
        .unwrap_or_default();

    if content_type.starts_with("multipart/form-data") {
        return handle_multipart_upload(req, tags_arc);
    }

    if content_type.starts_with("application/json") {
        let json: JsonPostBody = match rouille::input::json_input(req) {
            Ok(json) => json,
            Err(_) => {
                return api_response(400, false, "Invalid JSON body");
            }
        };

        if let Err(_) = process_text_submission(&json.text, tags_arc) {
            return api_response(500, false, "Failed to persist log message");
        }
        return api_response(200, true, "Log message accepted");
    }

    api_response(
        415,
        false,
        "Unsupported Content-Type. Use application/json or multipart/form-data",
    )
}

fn handle_multipart_upload(req: &Request, tags_arc: &Arc<Mutex<HashSet<String>>>) -> Response {
    let mut multipart = match rouille::input::multipart::get_multipart_input(req) {
        Ok(multipart) => multipart,
        Err(_) => {
            return api_response(400, false, "Malformed multipart request");
        }
    };

    let mut caption: Option<String> = None;
    let mut tags: Vec<String> = Vec::new();
    let mut upload_data: Option<UploadData> = None;

    while let Some(mut field) = multipart.next() {
        let field_name = field.headers.name.to_string();

        if field_name == "note" || field_name == "caption" {
            match read_limited_text(&mut field.data, MAX_NOTE_SIZE_BYTES) {
                Ok(value) => caption = Some(value),
                Err(message) => {
                    return api_response(400, false, &message);
                }
            }
            continue;
        }

        if field_name == "tags" {
            match read_limited_text(&mut field.data, MAX_NOTE_SIZE_BYTES) {
                Ok(value) => {
                    tags = value
                        .split(',')
                        .map(|tag| tag.trim().to_lowercase())
                        .filter(|tag| !tag.is_empty())
                        .collect();
                }
                Err(message) => {
                    return api_response(400, false, &message);
                }
            }
            continue;
        }

        if field_name == "file" || field_name == "image" || field_name == "audio" {
            if upload_data.is_some() {
                return api_response(400, false, "Only one file field is allowed");
            }

            let mime_type = match field.headers.content_type.as_ref() {
                Some(mime) => mime.essence_str().to_string(),
                None => {
                    return api_response(
                        415,
                        false,
                        "File Content-Type is required and must be an allowed image/audio MIME type",
                    );
                }
            };

            let media_type = match media_type_from_mime(&mime_type) {
                Some(media_type) => media_type,
                None => {
                    return api_response(
                        415,
                        false,
                        "Unsupported file MIME type. Allowed images: image/png, image/jpeg, image/webp, image/gif. Allowed audio: audio/mpeg, audio/wav, audio/ogg, audio/webm",
                    );
                }
            };

            match read_limited_binary(&mut field.data, MAX_UPLOAD_SIZE_BYTES) {
                Ok(bytes) => {
                    upload_data = Some(UploadData {
                        bytes,
                        mime_type,
                        media_type,
                    })
                }
                Err(ReadLimitError::TooLarge) => {
                    return api_response(413, false, "Uploaded file exceeds maximum size of 10 MB");
                }
                Err(ReadLimitError::Io) => {
                    return api_response(400, false, "Failed to read uploaded file");
                }
            }

            continue;
        }

        if read_limited_binary(&mut field.data, MAX_NOTE_SIZE_BYTES).is_err() {
            return api_response(400, false, "Failed to parse multipart field");
        }
    }

    let non_empty_caption = caption
        .as_ref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty());

    let Some(upload_data) = upload_data else {
        if let Some(caption_text) = non_empty_caption {
            if let Err(_) = process_text_submission(caption_text, tags_arc) {
                return api_response(500, false, "Failed to persist log message");
            }

            return api_response(201, true, "Text note accepted");
        }

        return api_response(
            400,
            false,
            "Missing note text or file field (file/image/audio)",
        );
    };

    let storage_result = persist_media_file(&upload_data);
    let stored_relative_path = match storage_result {
        Ok(path) => path,
        Err(_) => return api_response(500, false, "Failed to store uploaded media"),
    };

    if let Err(_) = super::write_media_log_entry(
        &stored_relative_path,
        &upload_data.mime_type,
        non_empty_caption,
        &tags,
    ) {
        let _ = fs::remove_file(&stored_relative_path);
        return api_response(500, false, "Failed to write log entry for uploaded media");
    }

    if let Some(caption_text) = non_empty_caption {
        for tag in super::find_tags(caption_text) {
            tags_arc.lock().unwrap().insert(tag);
        }
    }

    for tag in &tags {
        tags_arc.lock().unwrap().insert(tag.clone());
    }

    api_response(201, true, "Multipart upload accepted")
}

fn process_text_submission(
    text: &str,
    tags_arc: &Arc<Mutex<HashSet<String>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    super::write_log(text)?;

    let new_tags = super::find_tags(text);
    println!("New tags: {:?}", new_tags);
    for tag in new_tags {
        tags_arc.lock().unwrap().insert(tag);
    }

    Ok(())
}

fn api_response(status_code: u16, success: bool, message: &str) -> Response {
    Response::json(&ApiResponse {
        success,
        message: message.to_string(),
    })
    .with_status_code(status_code)
}

fn read_limited_text<R: Read>(reader: &mut R, max_bytes: usize) -> Result<String, String> {
    let mut data = Vec::new();
    let mut chunk = [0_u8; 8 * 1024];

    loop {
        let bytes_read = reader
            .read(&mut chunk)
            .map_err(|_| "Failed to read text field".to_string())?;
        if bytes_read == 0 {
            break;
        }

        data.extend_from_slice(&chunk[..bytes_read]);
        if data.len() > max_bytes {
            return Err(format!(
                "Text field exceeds maximum size of {} bytes",
                max_bytes
            ));
        }
    }

    String::from_utf8(data).map_err(|_| "Text field is not valid UTF-8".to_string())
}

struct UploadData {
    bytes: Vec<u8>,
    mime_type: String,
    media_type: MediaType,
}

enum MediaType {
    Image,
    Audio,
}

impl MediaType {
    fn root_path(&self) -> &'static str {
        match self {
            MediaType::Image => super::IMAGES_PATH,
            MediaType::Audio => super::AUDIO_PATH,
        }
    }
}

fn media_type_from_mime(mime: &str) -> Option<MediaType> {
    if ALLOWED_IMAGE_MIME_TYPES.contains(&mime) {
        return Some(MediaType::Image);
    }

    if ALLOWED_AUDIO_MIME_TYPES.contains(&mime) {
        return Some(MediaType::Audio);
    }

    None
}

fn extension_from_mime(mime: &str) -> Option<&'static str> {
    match mime {
        "image/png" => Some("png"),
        "image/jpeg" => Some("jpg"),
        "image/webp" => Some("webp"),
        "image/gif" => Some("gif"),
        "audio/mpeg" => Some("mp3"),
        "audio/wav" => Some("wav"),
        "audio/ogg" => Some("ogg"),
        "audio/webm" => Some("webm"),
        _ => None,
    }
}

fn persist_media_file(upload_data: &UploadData) -> Result<String, Box<dyn std::error::Error>> {
    let extension = extension_from_mime(&upload_data.mime_type)
        .ok_or("Unsupported MIME type for extension mapping")?;
    let now = Utc::now();
    let subdir = format!(
        "{}/{:04}/{:02}",
        upload_data.media_type.root_path(),
        now.year(),
        now.month()
    );

    fs::create_dir_all(&subdir)?;

    for _ in 0..10 {
        let filename = format!(
            "{}-{}.{}",
            now.timestamp_millis(),
            rand::thread_rng()
                .sample_iter(Alphanumeric)
                .take(6)
                .map(char::from)
                .collect::<String>()
                .to_lowercase(),
            extension
        );

        let mut full_path = PathBuf::from(&subdir);
        full_path.push(filename);

        if full_path.exists() {
            continue;
        }

        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&full_path)?;

        file.write_all(&upload_data.bytes)?;
        return Ok(full_path.to_string_lossy().to_string());
    }

    Err("Failed to generate unique file path".into())
}

enum ReadLimitError {
    TooLarge,
    Io,
}

fn read_limited_binary<R: Read>(
    reader: &mut R,
    max_bytes: usize,
) -> Result<Vec<u8>, ReadLimitError> {
    let mut data = Vec::new();
    let mut chunk = [0_u8; 8 * 1024];

    loop {
        let bytes_read = reader.read(&mut chunk).map_err(|_| ReadLimitError::Io)?;
        if bytes_read == 0 {
            break;
        }

        data.extend_from_slice(&chunk[..bytes_read]);
        if data.len() > max_bytes {
            return Err(ReadLimitError::TooLarge);
        }
    }

    Ok(data)
}
