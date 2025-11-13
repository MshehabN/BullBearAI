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
            '<h3>Prediction Results</h3>' +
            '<p><strong>Direction:</strong> <span style="color: ' + data.direction_color + '">' + data.direction + '</span></p>' +
            '<p><strong>Current Price:</strong> $' + data.current_price + '</p>' +
            '<p><strong>Predicted Price:</strong> $' + data.predicted_price + '</p>' +
            '<p><strong>Price Change:</strong> <span style="color: ' + data.change_color + '">$' + data.price_change + 
            ' (' + data.price_change_pct + '%)</span></p>' +
            '<p><strong>Confidence:</strong> ' + data.confidence + '%</p>';
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('predictionResults').innerHTML = '<div class="error">Error getting prediction</div>';
    });
}

