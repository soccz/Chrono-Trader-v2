// i18n (Internationalization) Handler

let currentLanguage = 'ko';
let translations = {};

// Load language file
async function loadLanguage(lang) {
    try {
        const response = await fetch(`/static/lang/${lang}.json`);
        translations = await response.json();
        currentLanguage = lang;
        applyTranslations();
        localStorage.setItem('language', lang);
        updateLanguageButtons();
    } catch (error) {
        console.error(`Error loading language ${lang}:`, error);
    }
}

// Apply translations to DOM elements
function applyTranslations() {
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        const translation = getNestedTranslation(key);

        if (translation) {
            if (element.tagName === 'INPUT' && element.type === 'text') {
                element.setAttribute('placeholder', translation);
            } else {
                element.textContent = translation;
            }
        }
    });
}

// Get nested translation value
function getNestedTranslation(key) {
    const keys = key.split('.');
    let value = translations;

    for (const k of keys) {
        if (value && typeof value === 'object' && k in value) {
            value = value[k];
        } else {
            return null;
        }
    }

    return value;
}

// Switch language
function switchLanguage(lang) {
    loadLanguage(lang);
}

// Update language button states
function updateLanguageButtons() {
    document.querySelectorAll('.btn-group button').forEach(btn => {
        btn.classList.remove('active');
    });

    const activeBtn = document.querySelector(`.btn-group button[onclick="switchLanguage('${currentLanguage}')"]`);
    if (activeBtn) {
        activeBtn.classList.add('active');
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function () {
    // Get saved language or default to Korean
    const savedLang = localStorage.getItem('language') || 'ko';
    loadLanguage(savedLang);
});
