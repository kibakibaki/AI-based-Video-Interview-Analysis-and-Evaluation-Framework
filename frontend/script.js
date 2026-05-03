const form = document.querySelector("#upload-form");
const input = document.querySelector("#video-input");
const selectedFile = document.querySelector("#selected-file");
const uploadButton = document.querySelector("#upload-button");
const statusBox = document.querySelector("#status");
const statusMessage = document.querySelector("#status-message");
const resultDetails = document.querySelector("#result-details");

let processingTimer = null;
let processingStartTime = null;

function setStatus(type, message, details = null) {
  statusBox.dataset.state = type;
  statusMessage.textContent = message;
  resultDetails.innerHTML = "";

  if (!details) {
    resultDetails.hidden = true;
    return;
  }

  Object.entries(details).forEach(([label, value]) => {
    const term = document.createElement("dt");
    term.textContent = label;

    const description = document.createElement("dd");
    description.textContent = value;

    resultDetails.append(term, description);
  });

  resultDetails.hidden = false;
}

function formatElapsedTime(totalSeconds) {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function startProcessingTimer(filename) {
  stopProcessingTimer();
  processingStartTime = Date.now();

  const updateTimer = () => {
    const elapsedSeconds = Math.floor((Date.now() - processingStartTime) / 1000);
    setStatus("loading", `Upload successful. Processing video... ${formatElapsedTime(elapsedSeconds)}`, {
      File: filename,
      "Processing time": formatElapsedTime(elapsedSeconds),
    });
  };

  updateTimer();
  processingTimer = window.setInterval(updateTimer, 1000);
}

function stopProcessingTimer() {
  if (processingTimer) {
    window.clearInterval(processingTimer);
    processingTimer = null;
  }
}

function getProcessingElapsedLabel() {
  if (!processingStartTime) {
    return "00:00";
  }

  const elapsedSeconds = Math.floor((Date.now() - processingStartTime) / 1000);
  return formatElapsedTime(elapsedSeconds);
}

input.addEventListener("change", () => {
  const file = input.files[0];
  selectedFile.textContent = file ? file.name : "No file selected";
  setStatus("idle", "Waiting for upload.");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = input.files[0];
  if (!file) {
    setStatus("error", "Please choose a video file first.");
    return;
  }

  const formData = new FormData();
  formData.append("video", file);

  uploadButton.disabled = true;
  uploadButton.textContent = "Uploading...";
  setStatus("loading", "Uploading and validating video...");

  try {
    const uploadResponse = await fetch("/upload", {
      method: "POST",
      body: formData,
    });
    const uploadResult = await uploadResponse.json();

    if (!uploadResponse.ok || !uploadResult.ok) {
      setStatus("error", uploadResult.error || "Upload failed.", {
        Duration: uploadResult.duration_label || "Unknown",
      });
      return;
    }

    startProcessingTimer(uploadResult.filename);
    uploadButton.textContent = "Processing...";

    const analyseResponse = await fetch("/analyse", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        filename: uploadResult.filename,
      }),
    });
    const analyseResult = await analyseResponse.json();

    stopProcessingTimer();

    if (!analyseResponse.ok || !analyseResult.ok) {
      setStatus("error", analyseResult.error || "Analysis failed.", {
        File: uploadResult.filename,
        "Processing time": getProcessingElapsedLabel(),
      });
      return;
    }

    const features = analyseResult.analysis.confidence_report.features;
    setStatus("success", "Processing completed.", {
      File: analyseResult.filename,
      Duration: analyseResult.analysis.duration_label,
      Saved: uploadResult.path,
      "Processing time": getProcessingElapsedLabel(),
      "Eye contact score": `${analyseResult.analysis.confidence_report.overall_score}/100`,
      "Confidence label": analyseResult.analysis.confidence_report.label,
      "Looking time": `${analyseResult.analysis.looking_total_time}s`,
      "Looking segments": analyseResult.analysis.segments.length,
      "Gaze center ratio": features.gaze_center_ratio,
      "Looking away time": `${features.looking_away_total_time}s`,
      "Longest away": `${features.longest_looking_away_duration}s`,
      "Blink rate": `${features.blink_rate_per_minute}/min`,
      "Head stability": features.head_movement_stability_score,
      "Annotated video": analyseResult.analysis.annotated_video,
      "Next step": analyseResult.next_step,
    });
    window.alert("Processing completed.");
  } catch (error) {
    stopProcessingTimer();
    setStatus("error", "Could not connect to the upload server.");
  } finally {
    uploadButton.disabled = false;
    uploadButton.textContent = "Upload Video";
  }
});
