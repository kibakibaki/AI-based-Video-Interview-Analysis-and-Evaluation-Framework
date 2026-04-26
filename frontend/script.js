const form = document.querySelector("#upload-form");
const input = document.querySelector("#video-input");
const selectedFile = document.querySelector("#selected-file");
const uploadButton = document.querySelector("#upload-button");
const statusBox = document.querySelector("#status");
const statusMessage = document.querySelector("#status-message");
const resultDetails = document.querySelector("#result-details");

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
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (!response.ok || !result.ok) {
      setStatus("error", result.error || "Upload failed.", {
        Duration: result.duration_label || "Unknown",
      });
      return;
    }

    setStatus("success", result.message, {
      File: result.filename,
      Duration: result.duration_label,
      Saved: result.path,
      "Next step": result.next_step,
    });
  } catch (error) {
    setStatus("error", "Could not connect to the upload server.");
  } finally {
    uploadButton.disabled = false;
    uploadButton.textContent = "Upload Video";
  }
});
