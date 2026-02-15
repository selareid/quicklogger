use std::collections::HashSet;

pub(crate) fn get_history_html(all_hists: HashSet<(String, String)>) -> String {
    let mut all_hists = Vec::from_iter(all_hists.iter());
    all_hists.sort_by_key(|&p| std::cmp::Reverse(p));

    let month_sections = all_hists
        .iter()
        .map(|(period, raw_logs)| {
            let mut lines: Vec<&str> = raw_logs.lines().collect();
            lines.reverse();

            let entries = lines
                .iter()
                .map(|line| format!("<div class=\"entry\">{}</div>", render_history_line(line)))
                .collect::<Vec<String>>()
                .join("\n");

            format!(
                "<section><h2>{}</h2><div class=\"entries\">{}</div></section>",
                escape_html(period),
                entries
            )
        })
        .collect::<Vec<String>>()
        .join("\n");

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QuickLogger History</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 0 auto; max-width: 860px; padding: 20px; background: #f5f5f5; color: #222; }}
a {{ color: #1a73e8; }}
section {{ margin-bottom: 22px; }}
h1 {{ margin-top: 0; }}
.entries {{ display: grid; gap: 12px; }}
.entry {{ background: #fff; border-radius: 8px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
.entry time {{ font-size: 12px; color: #555; }}
.entry .caption {{ margin: 8px 0; white-space: pre-wrap; }}
.entry .raw {{ white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
.entry img {{ max-width: 100%; border-radius: 6px; margin-top: 8px; }}
.entry audio {{ width: 100%; margin-top: 8px; }}
</style>
</head>
<body>
<h1>Log history</h1>
<p><a href="/">Back to logger</a></p>
{month_sections}
</body>
</html>"#
    )
}

fn render_history_line(line: &str) -> String {
    let Some((prefix, payload)) = line.split_once(": ") else {
        return format!("<div class=\"raw\">{}</div>", escape_html(line));
    };

    if !payload.starts_with("media ") {
        return format!(
            "<time>{}</time><div class=\"raw\">{}</div>",
            escape_html(prefix),
            escape_html(payload)
        );
    }

    let path = extract_media_field(payload, "path").unwrap_or_default();
    let mime = extract_media_field(payload, "mime").unwrap_or_default();
    let caption = extract_media_field(payload, "caption").unwrap_or_default();
    let timestamp = extract_media_field(payload, "timestamp").unwrap_or_else(|| prefix.to_string());

    let media_html = if mime.starts_with("audio/") {
        format!(
            "<audio controls preload=\"none\"><source src=\"{}\" type=\"{}\">Your browser does not support audio playback.</audio>",
            escape_html_attr(&normalize_media_path(&path)),
            escape_html_attr(&mime)
        )
    } else if mime.starts_with("image/") {
        format!(
            "<img src=\"{}\" alt=\"Uploaded image\" loading=\"lazy\">",
            escape_html_attr(&normalize_media_path(&path))
        )
    } else {
        format!(
            "<a href=\"{}\">Download attachment</a>",
            escape_html_attr(&normalize_media_path(&path))
        )
    };

    let caption_html = if caption.is_empty() {
        String::new()
    } else {
        format!("<div class=\"caption\">{}</div>", escape_html(&caption))
    };

    format!(
        "<time>{}</time>{}<div class=\"raw\">{}</div>{}",
        escape_html(&timestamp),
        caption_html,
        escape_html(payload),
        media_html,
    )
}

fn extract_media_field(payload: &str, key: &str) -> Option<String> {
    let needle = format!("{key}=\"");
    let start = payload.find(&needle)? + needle.len();
    let rest = &payload[start..];
    let end = rest.find('"')?;
    Some(rest[..end].replace("\\\"", "\""))
}

fn normalize_media_path(path: &str) -> String {
    path.trim_start_matches('.').to_string()
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

fn escape_html_attr(value: &str) -> String {
    escape_html(value)
}
