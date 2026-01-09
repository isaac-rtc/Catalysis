// static/js/toxicity_map.js
async function getPrediction() {
    const smiles = document.getElementById('smiles-input').value;
    const loader = document.getElementById('loader');
    const resultDiv = document.getElementById('result');
    const probSpan = document.getElementById('prob-value');
    const mapImg = document.getElementById('map-image');

    if (!smiles) return alert("Please enter a SMILES string");

    // UI State
    loader.style.display = 'block';
    resultDiv.style.display = 'none';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ smiles: smiles })
        });

        const data = await response.json();

        if (data.error) {
            alert("Error: " + data.error);
        } else {
            // Update text
            probSpan.innerText = (data.toxicity_prob * 100).toFixed(2) + "%";
            
            // Display the Base64 image
            // The 'data:image/png;base64,' prefix tells the browser how to interpret the string
            mapImg.src = `data:image/png;base64,${data.image_data}`;
            
            resultDiv.style.display = 'block';
        }
    } catch (err) {
        console.error(err);
        alert("Failed to connect to the server.");
    } finally {
        loader.style.display = 'none';
    }
}
