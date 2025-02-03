// JavaScript to preview uploaded image
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');

imageInput.addEventListener('change', function () {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImg.src = e.target.result;
            imagePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    } else {
        previewImg.src = "";
        imagePreview.style.display = 'none';
    }
});

function speakResult() {
    const resultElement = document.getElementById("result");
    if (resultElement) {
        const resultText = resultElement.innerText.replace("Prediction:", "").replace("🔊 Hear Result", "").trim();
        const utterance = new SpeechSynthesisUtterance(resultText);
        utterance.lang = "en-US"; 
        utterance.rate = 1; 
        utterance.pitch = 1; 

        const voices = window.speechSynthesis.getVoices();
        const selectedVoice = voices.find(voice => voice.name === "Google US English"); 
        if (selectedVoice) {
            utterance.voice = selectedVoice;
        }
        window.speechSynthesis.speak(utterance);
    } else {
        console.error("Result element not found.");
    }
}