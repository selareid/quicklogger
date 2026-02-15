mod history_view;
mod post_handlers;

use chrono::{Datelike, Utc};
use history_view::get_history_html;
use post_handlers::handle_post_request;
use rouille::Response;
use std::{
    collections::HashSet,
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

const INDEX_HTML: &str = include_str!("web/index.html");
const AUDIO_RECORDER_JS: &str = include_str!("web/audio-recorder.js");
pub(crate) const LOGS_PATH: &str = "./logs";
pub(crate) const IMAGES_PATH: &str = "./images";
pub(crate) const AUDIO_PATH: &str = "./audio";

fn main() {
    let web_addr = "0.0.0.0:7489";

    let index_html: String = if cfg!(debug_assertions) {
        String::from(INDEX_HTML)
    } else {
        INDEX_HTML.replace("<script>eruda.init();</script>", "")
    };

    let tags_arc: Arc<Mutex<HashSet<String>>> =
        Arc::new(Mutex::new(load_tags().expect("Error while loading tags.")));

    println!("Starting server at {}", web_addr);

    rouille::start_server(web_addr, move |req| {
        println!("Got a request\n {:?}", req);

        rouille::router!(req,
            (GET) (/) => {
                Response::html(&index_html)
            },
            (GET) (/audio-recorder_js) => {
                Response::from_data("application/javascript", AUDIO_RECORDER_JS)
            },
            (GET) (/history) => {
                Response::html(get_history_html(
                    get_log_files().unwrap_or_else(|_| HashSet::from([("_".to_string(), "Failed to read history.".to_string())]))
                ))
            },
            (GET) (/history/{year: usize}/{month: usize}) => {
                let logs: HashSet<(String, String)> = get_log_files()
                    .unwrap_or_else(|_| HashSet::new())
                    .iter()
                    .filter_map(|(p, s)| if *p == format!("{year}_{month}") {
                        Some((String::from(p), String::from(s)))
                        } else {None})
                    .collect();
                Response::html(get_history_html(logs))
            },
            (GET) (/images/{year: String}/{month: String}/{filename: String}) => {
                serve_media_file(IMAGES_PATH, &year, &month, &filename, image_content_type(&filename))
            },
            (GET) (/audio/{year: String}/{month: String}/{filename: String}) => {
                serve_media_file(AUDIO_PATH, &year, &month, &filename, audio_content_type(&filename))
            },
            (GET) (/tags) => {
                let t_v: Vec<String> = tags_arc.lock().unwrap().iter().cloned().collect();
                Response::text(t_v.join(", "))
            },
            (POST) (/) => {
                handle_post_request(req, &tags_arc)
            },
            (POST) (/upload) => {
                handle_post_request(req, &tags_arc)
            },
           _ => Response::text(format!("Error 404\n{:?}", req)).with_status_code(404)
        )
    });
}

pub(crate) fn write_log(body: &str) -> Result<(), Box<dyn std::error::Error>> {
    let current_date = Utc::now();
    let month_year: String = format!("{}_{}", current_date.year(), current_date.month());

    append_to_file(
        &format!(
            "{} {} {} {} {}: {}\n",
            current_date.timestamp(),
            current_date.weekday(),
            current_date.day(),
            current_date.month(),
            current_date.year(),
            body
        ),
        &format!("{}/{}", LOGS_PATH, month_year),
    )
}

pub(crate) fn write_media_log_entry(
    media_path: &str,
    mime_type: &str,
    caption: Option<&str>,
    tags: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let timestamp = Utc::now().to_rfc3339();
    let caption = caption.unwrap_or("").replace('"', "\\\"");
    let tags_string = if tags.is_empty() {
        String::new()
    } else {
        tags.join(",")
    };

    write_log(&format!(
        "media path=\"{}\" timestamp=\"{}\" mime=\"{}\" caption=\"{}\" tags=\"{}\"",
        media_path, timestamp, mime_type, caption, tags_string
    ))
}

fn append_to_file(body: &str, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent_dir) = std::path::Path::new(file_path).parent() {
        fs::create_dir_all(parent_dir)?;
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(file_path)?;

    file.write_all(body.as_bytes())?;
    Ok(())
}

fn get_log_files() -> Result<HashSet<(String, String)>, Box<dyn std::error::Error>> {
    let mut logfile_contents = HashSet::new();
    let paths = fs::read_dir(LOGS_PATH)?;

    for path_result in paths {
        let dir_entr = path_result?;
        let path = dir_entr.path();

        if path.is_file() {
            let file = OpenOptions::new().read(true).open(&path)?;
            let mut contents = String::new();
            let mut buffered_reader = std::io::BufReader::new(file);
            buffered_reader.read_to_string(&mut contents)?;

            if !contents.is_empty() {
                let filename = dir_entr.file_name().into_string().unwrap_or_default();
                logfile_contents.insert((filename, contents));
            }
        }
    }

    Ok(logfile_contents)
}

fn serve_media_file(
    media_root: &str,
    year: &str,
    month: &str,
    filename: &str,
    content_type: &'static str,
) -> Response {
    if [year, month, filename]
        .iter()
        .any(|part| part.contains('/') || part.contains("..") || part.contains('\\'))
    {
        return Response::empty_404();
    }

    let full_path = PathBuf::from(media_root)
        .join(year)
        .join(month)
        .join(filename);

    if !full_path.exists() || !full_path.is_file() {
        return Response::empty_404();
    }

    let canonical_root = match fs::canonicalize(media_root) {
        Ok(path) => path,
        Err(_) => return Response::empty_404(),
    };
    let canonical_file = match fs::canonicalize(&full_path) {
        Ok(path) => path,
        Err(_) => return Response::empty_404(),
    };

    if !canonical_file.starts_with(&canonical_root) {
        return Response::empty_404();
    }

    match fs::read(&full_path) {
        Ok(bytes) => Response::from_data(content_type, bytes),
        Err(_) => Response::empty_404(),
    }
}

fn image_content_type(filename: &str) -> &'static str {
    match Path::new(filename)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "webp" => "image/webp",
        "gif" => "image/gif",
        _ => "application/octet-stream",
    }
}

fn audio_content_type(filename: &str) -> &'static str {
    match Path::new(filename)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "webm" => "audio/webm",
        _ => "application/octet-stream",
    }
}

fn load_tags() -> Result<HashSet<String>, Box<dyn std::error::Error>> {
    let mut tags = HashSet::new();

    for (_, log_contents) in get_log_files()? {
        find_tags(&log_contents).iter().for_each(|t| {
            tags.insert(t.clone());
        });
    }

    Ok(tags)
}

pub(crate) fn find_tags(file_contents: &str) -> HashSet<String> {
    let mut tags: HashSet<String> = HashSet::new();

    for line in file_contents.lines() {
        let mut buf = String::new();

        for c in line.chars() {
            match c {
                '{' => {
                    if buf.is_empty() {
                        buf.push('{')
                    } else {
                        break;
                    }
                }
                '}' => {
                    if !buf.is_empty() {
                        buf.push('}');
                        tags.insert(buf.to_lowercase());
                        buf.clear();
                    }
                }
                _ => {
                    if !buf.is_empty() {
                        buf.push(c);
                    }
                }
            }
        }
    }

    tags
}
