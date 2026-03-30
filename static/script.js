document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("news-form");
    const textarea = document.getElementById("news-text");
    const submitBtn = document.getElementById("submit-btn");
    
    const errorContainer = document.getElementById("error-container");
    const loadingContainer = document.getElementById("loading");
    
    const resultContainer = document.getElementById("result-container");
    const confidenceBox = document.getElementById("confidence-box");
    const summarySection = document.getElementById("summary-section");
    const summaryBox = document.getElementById("summary-box");
    const fakeWarning = document.getElementById("fake-warning");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const newsText = textarea.value.trim();
        if (!newsText) {
            showError("Please enter some text to analyze.");
            return;
        }

        // Reset UI
        hideError();
        hideResult();
        showLoading();
        submitBtn.disabled = true;

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ news_text: newsText })
            });

            const data = await response.json();

            if (!response.ok) {
                showError(data.error || "An error occurred during analysis.");
                return;
            }

            renderResult(data.result);

        } catch (error) {
            console.error("Error:", error);
            showError("Failed to connect to the server. Please try again.");
        } finally {
            hideLoading();
            submitBtn.disabled = false;
        }
    });

    function showError(message) {
        errorContainer.textContent = message;
        errorContainer.style.display = "block";
    }

    function hideError() {
        errorContainer.style.display = "none";
        errorContainer.textContent = "";
    }

    function showLoading() {
        loadingContainer.style.display = "block";
    }

    function hideLoading() {
        loadingContainer.style.display = "none";
    }

    function hideResult() {
        resultContainer.style.display = "none";
        summarySection.style.display = "none";
        fakeWarning.style.display = "none";
        resultContainer.className = "result-box";
        confidenceBox.style.color = "";
    }

    function renderResult(result) {
        resultContainer.style.display = "block";

        if (result.is_real) {
            resultContainer.classList.add("real-news");
            confidenceBox.style.color = "var(--text-color)";
            confidenceBox.textContent = `TRUE NEWS (Confidence: ${result.confidence}%)`;
            
            summarySection.style.display = "block";
            summaryBox.textContent = result.summary;
        } else {
            resultContainer.classList.add("fake-news");
            confidenceBox.style.color = "#666";
            confidenceBox.textContent = `FAKE NEWS DETECTED (Confidence: ${result.confidence}%)`;
            
            fakeWarning.style.display = "block";
        }
    }
});
