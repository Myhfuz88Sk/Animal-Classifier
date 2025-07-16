document.getElementById('imageInput').addEventListener('change', function(event) {
    const reader = new FileReader();
    reader.onload = function(){
        const preview = document.getElementById('previewImage');
        preview.src = reader.result;
        preview.style.display = 'block';
    };
    reader.readAsDataURL(event.target.files[0]);
});
