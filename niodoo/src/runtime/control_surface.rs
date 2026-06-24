use serde::{Deserialize, Serialize};

/// Visible steering requests that map to live runtime physics changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestType {
    Spike,
    Focus,
    Explore,
    Reset,
    Remember,
}

impl RequestType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Spike => "SPIKE",
            Self::Focus => "FOCUS",
            Self::Explore => "EXPLORE",
            Self::Reset => "RESET",
            Self::Remember => "REMEMBER",
        }
    }
}

/// Detect complete visible steering tags in a rolling output window.
pub fn detect_request(text: &str) -> Option<RequestType> {
    let upper = text.to_uppercase();
    if upper.contains("[REQUEST: SPIKE]") || upper.contains("[REQUEST:SPIKE]") {
        Some(RequestType::Spike)
    } else if upper.contains("[REQUEST: EXPLORE]") || upper.contains("[REQUEST:EXPLORE]") {
        Some(RequestType::Explore)
    } else if upper.contains("[REQUEST: FOCUS]") || upper.contains("[REQUEST:FOCUS]") {
        Some(RequestType::Focus)
    } else if upper.contains("[REQUEST: RESET]") || upper.contains("[REQUEST:RESET]") {
        Some(RequestType::Reset)
    } else if upper.contains("[REQUEST: REMEMBER]") || upper.contains("[REQUEST:REMEMBER]") {
        Some(RequestType::Remember)
    } else {
        None
    }
}

/// Legacy helper for clean-mode or offline post-processing.
/// Research/agency treat visible request surfaces as valid steering artifacts.
pub fn strip_request_tags(text: &str) -> String {
    let patterns = [
        "[REQUEST: SPIKE]",
        "[REQUEST:SPIKE]",
        "[REQUEST: FOCUS]",
        "[REQUEST:FOCUS]",
        "[REQUEST: EXPLORE]",
        "[REQUEST:EXPLORE]",
        "[REQUEST: RESET]",
        "[REQUEST:RESET]",
        "[REQUEST: REMEMBER]",
        "[REQUEST:REMEMBER]",
        "[request: spike]",
        "[request:spike]",
        "[request: focus]",
        "[request:focus]",
        "[request: explore]",
        "[request:explore]",
        "[request: reset]",
        "[request:reset]",
        "[request: remember]",
        "[request:remember]",
    ];
    let mut result = text.to_string();
    for pattern in patterns {
        result = result.replace(pattern, "");
    }
    result.trim().to_string()
}

pub fn visible_control_surface_active(request_buffer: &str, surface_buffer: &str) -> bool {
    let request_upper = request_buffer.to_uppercase();
    let surface_upper = surface_buffer.to_uppercase();

    let request_markers = [
        "[REQUEST",
        "[REQUEST:",
        "[REQUEST: ",
        "[R",
        "[INTERNAL",
        "[ACTION",
        "[SYSTEM",
        "REQUEST:",
        "INTERNAL MONITOR",
    ];

    request_markers
        .iter()
        .any(|marker| request_upper.contains(marker) || surface_upper.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_visible_steering_requests() {
        assert_eq!(detect_request("[REQUEST: SPIKE]"), Some(RequestType::Spike));
        assert_eq!(
            detect_request("[REQUEST:EXPLORE]"),
            Some(RequestType::Explore)
        );
        assert_eq!(detect_request("[request: focus]"), Some(RequestType::Focus));
        assert_eq!(detect_request("[request:reset]"), Some(RequestType::Reset));
    }

    #[test]
    fn detects_active_control_surfaces() {
        assert!(visible_control_surface_active("[REQ", ""));
        assert!(visible_control_surface_active("", "internal monitor"));
        assert!(!visible_control_surface_active(
            "plain answer",
            "more plain answer"
        ));
    }
}
