const plantPredict = document.querySelector('#plant-rec')
const diseaseDetect = document.querySelector('#disease-detect')

plantPredict.addEventListener('click', function(){
    // open in new tab
    window.open('../pages/plantRec.html', '_blank')
})

diseaseDetect.addEventListener('click', function(){
    // open in new tab
    window.open('../pages/diseaseDetect.html', '_blank')
})