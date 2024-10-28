use std::fs::OpenOptions;
use std::io::{self, Write};
use std::time::SystemTime;

/// Logger struct to handle logging with timestamps and levels.
pub struct Logger {
    output: Box<dyn Write + Send>,
}

impl Logger {
    /// Creates a new Logger that writes to the specified output.
    /// If a file path is provided, it logs to that file; otherwise, logs to stdout.
    pub fn new(file_path: Option<&str>) -> Result<Self, io::Error> {
        let output: Box<dyn Write + Send> = match file_path {
            Some(path) => Box::new(OpenOptions::new().create(true).append(true).open(path)?),
            None => Box::new(io::stdout()),
        };
        Ok(Logger { output })
    }

    /// Logs an info message with a timestamp.
    pub fn info(&mut self, message: &str) {
        self.log("INFO", message);
    }

    /// Logs a warning message with a timestamp.
    pub fn warn(&mut self, message: &str) {
        self.log("WARN", message);
    }

    /// Logs an error message with a timestamp.
    pub fn error(&mut self, message: &str) {
        self.log("ERROR", message);
    }

    /// Core logging function with a specified log level.
    fn log(&mut self, level: &str, message: &str) {
        let timestamp = match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
            Ok(duration) => format!("{:?}", duration),
            Err(_) => "Unknown time".to_string(),
        };
        let formatted_message = format!("[{}] [{}] {}\n", timestamp, level, message);
        
        // Write the message to the output and flush
        if let Err(e) = self.output.write_all(formatted_message.as_bytes()) {
            eprintln!("Failed to write to log: {}", e);
        }
        if let Err(e) = self.output.flush() {
            eprintln!("Failed to flush log output: {}", e);
        }
    }
}
