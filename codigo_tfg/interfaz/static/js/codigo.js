// Función para manejar el cambio en el input file
function handleFileInput(inputId, outputId) {
    const fileInput = document.getElementById(inputId);         // Elemento input file
    const outputElement = document.getElementById(outputId);    // Elemento para mostrar archivos seleccionados

    // Espera el evento para campo input file
    fileInput.addEventListener('change', function(event) {
        const files = event.target.files;   // Obtener los archivos seleccionados

        if (files.length > 0) {
            const fileNames = [];

            for (let i = 0; i < files.length; i++) {
                const file = files[i];

                if (file.type === "application/json") {     // Verificar si el archivo es de tipo JSON
                    fileNames.push(file.name);
                } else {
                    console.log("Por favor, selecciona solo archivos JSON.");
                    alert("Por favor, selecciona solo archivos JSON.");

                    event.target.value = "";    // Limpiar el valor del input file

                    return;
                }
            }

            const selectedFilesElement = "<u>Archivos seleccionados</u>: <i>" + fileNames.join(", ") + "</i>";

            // Actualiza el contenido del elemento HTML con el texto seleccionado
            outputElement.innerHTML = selectedFilesElement;
        } else {
            console.log("Por favor, selecciona al menos un archivo.");
        }
    });
}


// Función para verificar si se han seleccionado archivos en los inputs
function checkFileInputs() {
    const fileInput1 = document.getElementById('fileInput1');
    const fileInput2 = document.getElementById('fileInput2');
    const fileInput3 = document.getElementById('fileInput3');

    if (fileInput1.files.length === 0 && fileInput2.files.length === 0 && fileInput3.files.length === 0) {  // Comprobar si ficheros
        alert("Por favor, selecciona al menos un archivo antes de enviar.");
        return false;  // Prevenir el envío del formulario
    }
    
    return true;  // Permitir el envío del formulario
}



// Manejar los tres inputs de archivo
handleFileInput('fileInput1', 'selectedFiles1');
handleFileInput('fileInput2', 'selectedFiles2');
handleFileInput('fileInput3', 'selectedFiles3');
