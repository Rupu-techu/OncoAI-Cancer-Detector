const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('image-input');
const form = document.getElementById('upload-form');
const spinner = document.getElementById('spinner');
const previewImg = document.getElementById('preview-img');

if (dropZone && fileInput) {
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-primary');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('border-primary'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-primary');
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            showPreview(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            showPreview(e.target.files[0]);
        }
    });
}

function showPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewImg.classList.remove('d-none');
    };
    reader.readAsDataURL(file);
}

if (form && spinner) {
    form.addEventListener('submit', () => {
        spinner.classList.remove('d-none');
    });
}
