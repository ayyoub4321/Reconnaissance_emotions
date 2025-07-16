document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const browseButton = document.getElementById('browse-button');
    const fileInfoSpan = document.querySelector('.file-info span');
    const uploadContainer = document.querySelector('.upload-container');

    // Lier le bouton "Parcourir..." à l'input file
    browseButton.addEventListener('click', () => {
        fileInput.click();
    });

    // Fonction pour afficher l'image sélectionnée
    function showSelectedImage(file) {
        const reader = new FileReader();

        reader.onload = function(e) {
            // Supprimer toute image précédente
            const existingImg = document.getElementById('preview-img');
            if (existingImg) {
                existingImg.remove();
            }

            // Créer et insérer la nouvelle image
            const img = document.getElementById('img');
            
            id = 'preview-img';
            src = e.target.result;
            img.innerHTML='<img  src="'+src+'" class="imgSlect"/>'
        };

        reader.readAsDataURL(file);
    }

    // Mettre à jour l'affichage du nom du fichier sélectionné
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            fileInfoSpan.textContent = fileInput.files[0].name;
            showSelectedImage(fileInput.files[0]);
        } else {
            fileInfoSpan.textContent = 'No file selected';
        }
    });

    // Gérer le drag and drop
    const dropZone = document.querySelector('.drop-zone');
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#007bff';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#ccc';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#ccc';

        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile && droppedFile.type.startsWith('image/')) {
            fileInput.files = [droppedFile]; // Affecter correctement les fichiers
            fileInfoSpan.textContent = droppedFile.name;
            showSelectedImage(droppedFile);
        }
    });
});