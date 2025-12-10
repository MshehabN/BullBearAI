let currentSymbol = '';

function searchStock() {
    // get the symbol from input and make it uppercase
    const symbol = document.getElementById('symbolInput').value.trim().toUpperCase();
    
    if (!symbol) {
        alert('Please enter a stock symbol');
        return;
    }
    
    currentSymbol = symbol;
    
    // send request to backend
    fetch('/api/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ symbol: symbol })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        
        // display the stock info
        document.getElementById('stockSymbol').textContent = data.symbol;
        document.getElementById('currentPrice').textContent = data.price;
        document.getElementById('stockInfo').style.display = 'block';
        
        // show chart if available
        if (data.chart) {
            const chartImg = document.createElement('img');
            chartImg.src = 'data:image/png;base64,' + data.chart;
            chartImg.style.width = '100%';
            chartImg.style.maxWidth = '800px';
            document.getElementById('chartContainer').innerHTML = '';
            document.getElementById('chartContainer').appendChild(chartImg);
        }
        
        // show prediction section
        document.getElementById('predictionSection').style.display = 'block';
        document.getElementById('predictionResults').innerHTML = '';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error searching for stock');
    });
}


function getPrediction() {
    if (!currentSymbol) {
        alert('Please search for a stock first');
        return;
    }
    
    // show loading message
    document.getElementById('predictionResults').innerHTML = '<p>Getting prediction... (this may take a moment)</p>';
    
    // request prediction from backend
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ symbol: currentSymbol })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('predictionResults').innerHTML = '<div class="error">Error: ' + data.error + '</div>';
            return;
        }
        
        // display the prediction results
        document.getElementById('predictionResults').innerHTML = 
            '<div class="prediction-header">' +
            '<h3>Prediction Results</h3>' +
            '<div class="direction-badge" style="background-color: ' + data.direction_color + '20; border-color: ' + data.direction_color + ';">' +
            '<span class="direction-label">Direction</span>' +
            '<span class="direction-value" style="color: ' + data.direction_color + ';">' + data.direction + '</span>' +
            '</div>' +
            '</div>' +
            '<div class="prediction-grid">' +
            '<div class="prediction-item">' +
            '<span class="prediction-label">Current Price</span>' +
            '<span class="prediction-value">$' + data.current_price + '</span>' +
            '</div>' +
            '<div class="prediction-item">' +
            '<span class="prediction-label">Predicted Price</span>' +
            '<span class="prediction-value">$' + data.predicted_price + '</span>' +
            '</div>' +
            '<div class="prediction-item">' +
            '<span class="prediction-label">Price Change</span>' +
            '<span class="prediction-value" style="color: ' + data.change_color + ';">$' + data.price_change + 
            ' (' + data.price_change_pct + '%)</span>' +
            '</div>' +
            '<div class="prediction-item">' +
            '<span class="prediction-label">Confidence</span>' +
            '<span class="prediction-value">' + data.confidence + '%</span>' +
            '</div>' +
            '</div>';
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('predictionResults').innerHTML = '<div class="error">Error getting prediction</div>';
    });
}

