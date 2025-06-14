use chrono::{Datelike, Utc};
use rouille::Response;
use serde::Deserialize;
use std::{
    collections::HashSet,
    fs::{self, OpenOptions},
    io::{Read, Write},
    sync::{Arc, Mutex},
};

const INDEX_HTML: &str = include_str!("web/index.html");
const LOGS_PATH: &str = "./logs";

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
            (GET) (/history) => {
                Response::text(get_history_string(
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
                Response::text(get_history_string(logs))
            },
            (GET) (/tags) => {
                // respond with list of tags
                let t_v: Vec<String> = tags_arc.lock().unwrap().iter().map(|s| s.clone()).collect();
                Response::text(t_v.join(", "))
            },
            (POST) (/) => {
                #[derive(Deserialize)]
                #[derive(Debug)]
                struct Json {
                    text: String,
                }

                let json: Json = rouille::try_or_400!(rouille::input::json_input(req));

                // Write to disk
                write_log(&json.text);

                // add new tags to tag list
                let new_tags = find_tags(&json.text);
                println!("New tags: {:?}", new_tags);
                for t in new_tags.into_iter() {
                    tags_arc.lock().unwrap().insert(t);
                }


                Response::text(format!("field's value is {}", json.text))
            },
           _ => Response::text(format!("Error 404\n{:?}", req)).with_status_code(404)
        )
    });
}

fn write_log(body: &String) {
    let current_date = Utc::now();
    let month_year: String = format!("{}_{}", current_date.year(), current_date.month());

    match append_to_file(
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
    ) {
        Ok(_) => (),
        Err(e) => eprintln!("Error when writing to file!\n{:?}", e),
    }
}

fn append_to_file(body: &str, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Open the file in append mode, create it if it doesn't exist
    let mut file = OpenOptions::new()
        .create(true) // Create the file if it doesn't exist
        .append(true) // Open in append mode
        .open(file_path)?; // Open the file

    // Write the body content to the file
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

            if contents.len() > 0 {
                let filename = dir_entr.file_name().into_string().unwrap_or_default();

                logfile_contents.insert((filename, contents));
            }
        }
    }

    Ok(logfile_contents)
}

fn get_history_string(all_hists: HashSet<(String, String)>) -> String {
    let mut all_hists = Vec::from_iter(all_hists.iter());
    all_hists.sort_by_key(|&p| std::cmp::Reverse(p)); // sort all_hists by date (filename)

    all_hists
        .iter()
        .map(|(p, s)| {
            let mut lines: Vec<String> = s.lines().map(|l| l.to_string()).collect();
            lines.reverse();
            let month_hist: String = lines.join("\n");
            format!("{p}:\n{month_hist}")
        })
        .collect::<Vec<String>>()
        .join("\n")
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

// find tags {<tag>} in file
fn find_tags(file_contents: &String) -> HashSet<String> {
    let mut tags: HashSet<String> = HashSet::new();

    for line in file_contents.lines() {
        // find tags
        let mut buf = String::new();

        for c in line.chars() {
            match c {
                '{' => {
                    if buf.len() == 0 {
                        buf.push('{')
                    } else {
                        // two '{' detected before closing '}' found
                        break; // quit, this line is a lost cause
                    }
                }
                '}' => {
                    if buf.len() > 0 {
                        // buf contains {...
                        buf.push('}');
                        tags.insert(buf.to_lowercase());
                        buf.clear();
                    }
                }
                _ => {
                    if buf.len() > 0 {
                        buf.push(c);
                    }
                }
            }
        }
    }

    tags
}
