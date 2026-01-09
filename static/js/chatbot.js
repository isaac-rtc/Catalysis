const chatbot = document.getElementById("chatbot");
const toggleBtn = document.getElementById("chatbot-toggle");
const responseBox = document.getElementById("chatbot-response");

toggleBtn.addEventListener("click", () => {
    chatbot.classList.toggle("collapsed");
    chatbot.classList.toggle("expanded");
});

async function fakeResponse(button) {
    if (button.disabled) return;

    /** @type {HTMLIFrameElement} */
    const sketcherFrame = document.querySelector('[data-sketcher]');
    const sketcherModule = sketcherFrame.contentWindow.Module;
    const smiles = sketcherModule.sketcher_export_text(sketcherModule.Format.SMILES);

    button.disabled = true;
    responseBox.innerText = "Loading...";

    try {
        const res = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ smiles })
        });

        const data = await res.json();
        responseBox.innerText = data.text;

    } catch (err) {
        responseBox.innerText = "Error connecting to backend.";
    } finally {
        button.disabled = false;
    }
}
