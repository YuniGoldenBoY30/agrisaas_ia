document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const resultDiv = document.getElementById('result');
    const originalImage = document.getElementById('original-image');
    const predictionImage = document.getElementById('prediction-image');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            const response = await axios.post('http://localhost:8000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });

            originalImage.src = `data:image/png;base64,${response.data.original_image}`;
            predictionImage.src = `data:image/png;base64,${response.data.prediction_image}`;
            resultDiv.classList.remove('hidden');
        } catch (error) {
            console.error('Error:', error);
            alert('Hubo un error al procesar la imagen. Por favor, intenta de nuevo.');
        }
    });
});