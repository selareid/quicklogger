use chrono::{Datelike, Utc};
use rouille::Response;
use serde::Deserialize;
use std::{fs::OpenOptions, io::Write};

const INDEX_HTML: &str = include_str!("web/index.html");
const LOGS_PATH: &str = "./logs";

fn main() {
    let web_addr = "0.0.0.0:7489";
    let ext_addr = "logger.selareid.xyz";

    let index_html = if cfg!(debug_assertions) { INDEX_HTML.replace("example.com", &ext_addr) }
                        else { INDEX_HTML.replace("example.com", &ext_addr).replace("<script>eruda.init();</script>", "") };

    rouille::start_server(web_addr, move |req| {
        println!("Got a request\n {:?}", req);

        rouille::router!(req,
            (GET) (/) => {
                // rouille::Response::text("Go away").with_status_code(200)
                Response::html(&index_html)
            },
            (POST) (/) => {
                // Response::empty_204()
                #[derive(Deserialize)]
                #[derive(Debug)]
                struct Json {
                    text: String,
                }

                let json: Json = rouille::try_or_400!(rouille::input::json_input(req));

                // Write to disk
                write_log(&json.text);


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
