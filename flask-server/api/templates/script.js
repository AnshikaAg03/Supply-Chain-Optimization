// Toggle Normal Analysis dropdown
const normalLink = document.getElementById('normal-analysis-link');
const normalDropdown = document.getElementById('normal-dropdown');

normalLink.addEventListener('click', function(e) {
    e.preventDefault(); // Prevent default link behavior
    normalDropdown.style.display = normalDropdown.style.display === 'block' ? 'none' : 'block';
});

// Toggle Advanced Analysis dropdown
const advancedLink = document.getElementById('advanced-analysis-link');
const advancedDropdown = document.getElementById('advanced-dropdown');

advancedLink.addEventListener('click', function(e) {
    e.preventDefault(); // Prevent default link behavior
    advancedDropdown.style.display = advancedDropdown.style.display === 'block' ? 'none' : 'block';
});

// Optionally, close dropdowns when clicking outside
window.addEventListener('click', function(e) {
    if (!normalLink.contains(e.target) && !normalDropdown.contains(e.target)) {
        normalDropdown.style.display = 'none';
    }
    if (!advancedLink.contains(e.target) && !advancedDropdown.contains(e.target)) {
        advancedDropdown.style.display = 'none';
    }
});
