import { renderSequence } from "./render-sequence.js";

async function main() {
  const urlParams = new URLSearchParams(window.location.search);

  const sequenceName = urlParams.get("dir");

  const sequenceNameDiv = document.querySelector(".sequence_name");

  sequenceNameDiv.textContent = sequenceName;

  const canvas = document.querySelector(".canvas1");

  const interval = setInterval(() => {
    const loadingDiv = document.querySelector(".is_loading_text");
    if (loadingDiv.textContent.includes("Loading..."))
      loadingDiv.textContent = "Loading.";
    else if (loadingDiv.textContent.includes("Loading.."))
      loadingDiv.textContent = "Loading...";
    else if (loadingDiv.textContent.includes("Loading."))
      loadingDiv.textContent = "Loading..";
  }, 300);

  const {
    totalFrames,
    onChangeFrame,
    updateCurrentFrame,
    cameraLabels,
    onChangeCamera,
  } = await renderSequence(sequenceName, canvas);

  const tags = [];
  tags.forEach((tag) => {
    const div = document.createElement("div");
    div.textContent = tag;
    div.className = `rounded-[4px] px-[4px] bg-blue-500`;
    document.querySelector(".tags").appendChild(div);
  });

  const cameraLabelDivs = cameraLabels.map((label) => {
    const div = document.createElement("div");

    div.textContent = label;
    document.querySelector(".camera_button_container").appendChild(div);

    return div;
  });

  cameraLabelDivs.forEach((div) => {
    const baseClassName = "rounded-[4px] px-[4px]";
    const unselectedClassName = `${baseClassName} bg-gray-200`;
    const selectedClassName = `${baseClassName} bg-green-700`;
    div.className =
      div.textContent === "default" ? selectedClassName : unselectedClassName;
    div.addEventListener("click", () => {
      cameraLabelDivs.forEach((div) => (div.className = unselectedClassName));
      div.className = selectedClassName;
      onChangeCamera(div.textContent);
    });
  });

  document.querySelector(".total_frames").textContent = totalFrames;

  document.querySelector(".is_loading").style.display = "none";
  clearInterval(interval);

  const progressBarContainer = document.querySelector(".progress-bar-bg");
  const progressBarFill = document.querySelector(".progress-bar-fill");
  const progressBarThumb = document.querySelector(".progress-bar-thumb");

  onChangeFrame((currentFrame) => {
    document.querySelector(".current_frame").textContent = currentFrame;
    const percentage = (currentFrame / totalFrames) * 100;
    progressBarFill.style.width = `${percentage}%`;
    progressBarThumb.style.left = `${percentage}%`;
  });

  let isDragging = false;

  progressBarThumb.addEventListener("mousedown", (e) => {
    isDragging = true;
    handleMouseEvent(e);
  });

  document.addEventListener("mousemove", (e) => {
    if (isDragging) {
      handleMouseEvent(e);
    }
  });

  document.addEventListener("mouseup", () => {
    isDragging = false;
  });

  progressBarContainer.addEventListener("click", (e) => {
    handleMouseEvent(e);
  });

  function handleMouseEvent(e) {
    const rect = progressBarContainer.getBoundingClientRect();
    const offsetX = Math.min(Math.max(0, e.clientX - rect.left), rect.width);
    const percentage = (offsetX / rect.width) * 100;

    updateCurrentFrame(Math.floor((percentage / 100) * totalFrames));
    progressBarFill.style.width = `${percentage}%`;
    progressBarThumb.style.left = `${percentage}%`;
  }
}

main();
