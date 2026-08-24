document.getElementById("predict-btn").addEventListener("click", async () => {
    const checked = document.querySelectorAll('input[name="driver"]:checked');
    const drivers = Array.from(checked).map(c => c.value);
    const circuit = document.getElementById("circuit-select").value;

    if (drivers.length === 0) {
        alert("Select at least one driver.");
        return;
    }

    const response = await fetch("/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ drivers, circuit, year: 2026 })
    });
    const data = await response.json();
    renderResults(data.predictions);
});

function renderResults(predictions) {
    const panel = document.getElementById("results-panel");
    panel.innerHTML = predictions.map(p => `
        <div class="prediction-card">
            <h3>${p.driver}</h3>
            <p>${p.predicted_delta_s}s</p>
            <p>${p.lower_bound_s}s – ${p.upper_bound_s}s</p>
        </div>
    `).join("");

    const trace = {
        x: predictions.map(p => p.driver),
        y: predictions.map(p => p.predicted_delta_s),
        error_y: {
            array: predictions.map(p => p.upper_bound_s - p.predicted_delta_s),
            arrayminus: predictions.map(p => p.predicted_delta_s - p.lower_bound_s),
        },
        type: "bar",
        marker: { color: "#e10600" },
    };

    Plotly.newPlot("shap-chart", [trace], {
        title: "Predicted Qualifying Delta by Driver",
        paper_bgcolor: "#1a1a26",
        plot_bgcolor: "#1a1a26",
        font: { color: "white" },
        yaxis: { title: "Delta to fastest (s)", gridcolor: "#2f2f48" },
    });
}