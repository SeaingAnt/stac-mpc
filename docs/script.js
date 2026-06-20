document.addEventListener("DOMContentLoaded", () => {
    const copyBtn = document.getElementById("copyBibtex");
    const bibtexCode = document.getElementById("bibtexCode");

    if (copyBtn && bibtexCode) {
        copyBtn.addEventListener("click", async () => {
            try {
                await navigator.clipboard.writeText(bibtexCode.innerText);
                const originalText = copyBtn.innerText;
                
                // Visual feedback for successful copy
                copyBtn.innerText = "Copied!";
                copyBtn.style.background = "rgba(16, 185, 129, 0.2)"; // emerald 500 with opacity
                copyBtn.style.borderColor = "rgba(16, 185, 129, 0.5)";
                
                // Reset button state after 2 seconds
                setTimeout(() => {
                    copyBtn.innerText = originalText;
                    copyBtn.style.background = "rgba(255, 255, 255, 0.1)";
                    copyBtn.style.borderColor = "rgba(255, 255, 255, 0.2)";
                }, 2000);
            } catch (err) {
                console.error("Failed to copy text: ", err);
                copyBtn.innerText = "Failed";
            }
        });
    }
});
