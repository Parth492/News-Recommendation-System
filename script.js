const container = document.querySelector('.container');

fetch('/api/recommend', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    'index': 19  // Replace with the index of the news article you want to use as input
  })
})
.then(response => response.json())
.then(data => {
  // Create HTML code for each news item and add it to the container
  data.news.forEach((headline, i) => {
    const div = document.createElement('div');
    div.classList.add('news-item');
    div.innerHTML = `
      <h2>${headline}</h2>
      <p>Index: ${data.index[i]}</p>
    `;
    container.appendChild(div);
    
  });
});
