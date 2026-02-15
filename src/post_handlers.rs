use rouille::{Request, Response};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    io::Read,
    sync::{Arc, Mutex},
};

const MAX_UPLOAD_SIZE_BYTES: usize = 10 * 1024 * 1024;
const MAX_NOTE_SIZE_BYTES: usize = 64 * 1024;
const ALLOWED_IMAGE_MIME_TYPES: [&str; 4] = ["image/png", "image/jpeg", "image/webp", "image/gif"];

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

        process_text_submission(&json.text, tags_arc);
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

    let mut note: Option<String> = None;
    let mut has_uploaded_file = false;

    while let Some(mut field) = multipart.next() {
        let field_name = field.headers.name.to_string();

        if field_name == "note" || field_name == "caption" {
            match read_limited_text(&mut field.data, MAX_NOTE_SIZE_BYTES) {
                Ok(value) => note = Some(value),
                Err(message) => {
                    return api_response(400, false, &message);
                }
            }
            continue;
        }

        if field_name == "file" || field_name == "image" {
            if has_uploaded_file {
                return api_response(400, false, "Only one file field is allowed");
            }

            let mime_type = match field.headers.content_type.as_ref() {
                Some(mime) => mime.essence_str().to_string(),
                None => {
                    return api_response(
                        415,
                        false,
                        "File Content-Type is required and must be an allowed image MIME type",
                    );
                }
            };

            if !ALLOWED_IMAGE_MIME_TYPES.contains(&mime_type.as_str()) {
                return api_response(
                    415,
                    false,
                    "Unsupported file MIME type. Allowed: image/png, image/jpeg, image/webp, image/gif",
                );
            }

            match read_limited_binary_len(&mut field.data, MAX_UPLOAD_SIZE_BYTES) {
                Ok(_) => has_uploaded_file = true,
                Err(ReadLimitError::TooLarge) => {
                    return api_response(413, false, "Uploaded file exceeds maximum size of 10 MB");
                }
                Err(ReadLimitError::Io) => {
                    return api_response(400, false, "Failed to read uploaded file");
                }
            }

            continue;
        }

        if read_limited_binary_len(&mut field.data, MAX_NOTE_SIZE_BYTES).is_err() {
            return api_response(400, false, "Failed to parse multipart field");
        }
    }

    if !has_uploaded_file {
        return api_response(400, false, "Missing required file field (file/image)");
    }

    if let Some(note_text) = note.as_ref() {
        if !note_text.trim().is_empty() {
            process_text_submission(note_text, tags_arc);
        }
    }

    api_response(201, true, "Multipart upload accepted")
}

fn process_text_submission(text: &str, tags_arc: &Arc<Mutex<HashSet<String>>>) {
    super::write_log(text);

    let new_tags = super::find_tags(text);
    println!("New tags: {:?}", new_tags);
    for tag in new_tags {
        tags_arc.lock().unwrap().insert(tag);
    }
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

enum ReadLimitError {
    TooLarge,
    Io,
}

fn read_limited_binary_len<R: Read>(
    reader: &mut R,
    max_bytes: usize,
) -> Result<usize, ReadLimitError> {
    let mut total_bytes = 0_usize;
    let mut chunk = [0_u8; 8 * 1024];

    loop {
        let bytes_read = reader.read(&mut chunk).map_err(|_| ReadLimitError::Io)?;
        if bytes_read == 0 {
            break;
        }

        total_bytes += bytes_read;
        if total_bytes > max_bytes {
            return Err(ReadLimitError::TooLarge);
        }
    }

    Ok(total_bytes)
}
