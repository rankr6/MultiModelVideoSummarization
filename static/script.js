document.getElementById("summarize-btn").addEventListener("click", function() {
    const videoInput = document.getElementById("video-upload");
    const videoFile = videoInput.files[0];

    if (videoFile) {
        const formData = new FormData();
        formData.append("file", videoFile);

        // Send video file to server for upload and processing
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.summary) {
                // Show the video
                const videoPlayer = document.getElementById("video-player");
                const videoUrl = `/static/uploads/${data.filename}`;  // Assuming the filename is returned in the response
                videoPlayer.src = videoUrl;
                videoPlayer.style.display = "block";  // Ensure the video player is visible

                // Display the summary
                const summaryText = document.getElementById("summary-text");
                summaryText.textContent = data.summary;
            } else if (data.error) {
                alert("Error: " + data.error);
            }
        })
        .catch(error => {
            alert("Error: " + error);
        });
    } else {
        alert("Please upload a video first.");
    }
});
