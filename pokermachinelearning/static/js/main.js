// Main JavaScript file for Poker Range Classifier

// Utility functions
const Utils = {
    // Format numbers with commas
    formatNumber: (num) => {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    },
    
    // Format percentages
    formatPercent: (num) => {
        return (num * 100).toFixed(1) + '%';
    },
    
    // Show loading spinner
    showLoading: (element) => {
        element.innerHTML = '<div class="text-center"><i class="fas fa-spinner fa-spin fa-2x"></i><br><small>Loading...</small></div>';
    },
    
    // Show success message
    showSuccess: (element, message) => {
        element.innerHTML = `<div class="alert alert-success"><i class="fas fa-check-circle me-2"></i>${message}</div>`;
    },
    
    // Show error message
    showError: (element, message) => {
        element.innerHTML = `<div class="alert alert-danger"><i class="fas fa-exclamation-triangle me-2"></i>${message}</div>`;
    },
    
    // Animate element
    animateElement: (element, animation) => {
        element.classList.add(animation);
        setTimeout(() => {
            element.classList.remove(animation);
        }, 1000);
    },
    
    // Debounce function
    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};

// Chart utilities
const ChartUtils = {
    // Create probability chart
    createProbabilityChart: (canvasId, data, labels) => {
        const ctx = document.getElementById(canvasId).getContext('2d');
        return new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: [
                        '#dc3545', // Nuts - Red
                        '#fd7e14', // Strong - Orange
                        '#ffc107', // Marginal - Yellow
                        '#198754'  // Bluff - Green
                    ],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true,
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed;
                                return `${label}: ${Utils.formatPercent(value)}`;
                            }
                        }
                    }
                },
                animation: {
                    animateRotate: true,
                    animateScale: true
                }
            }
        });
    },
    
    // Create performance chart
    createPerformanceChart: (canvasId, data) => {
        const ctx = document.getElementById(canvasId).getContext('2d');
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Nuts', 'Strong', 'Marginal', 'Bluff'],
                datasets: [{
                    label: 'Accuracy',
                    data: data,
                    backgroundColor: [
                        'rgba(220, 53, 69, 0.8)',
                        'rgba(253, 126, 20, 0.8)',
                        'rgba(255, 193, 7, 0.8)',
                        'rgba(25, 135, 84, 0.8)'
                    ],
                    borderColor: [
                        '#dc3545',
                        '#fd7e14',
                        '#ffc107',
                        '#198754'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return Utils.formatPercent(value);
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Accuracy: ${Utils.formatPercent(context.parsed.y)}`;
                            }
                        }
                    }
                }
            }
        });
    }
};

// Form validation
const FormValidator = {
    // Validate hand classification form
    validateClassificationForm: (formData) => {
        const errors = [];
        
        if (!formData.position || formData.position === '') {
            errors.push('Position is required');
        }
        
        if (!formData.num_players || formData.num_players === '') {
            errors.push('Number of players is required');
        }
        
        if (!formData.pot_size || formData.pot_size <= 0) {
            errors.push('Pot size must be greater than 0');
        }
        
        if (!formData.stack_size || formData.stack_size <= 0) {
            errors.push('Stack size must be greater than 0');
        }
        
        if (!formData.actions || formData.actions.length === 0) {
            errors.push('At least one action is required');
        }
        
        return errors;
    },
    
    // Validate training form
    validateTrainingForm: (formData) => {
        const errors = [];
        
        if (!formData.num_hands || formData.num_hands <= 0) {
            errors.push('Number of hands must be greater than 0');
        }
        
        if (formData.num_hands > 50000) {
            errors.push('Number of hands cannot exceed 50,000');
        }
        
        return errors;
    }
};

// API client
const API = {
    // Make API request
    request: async (url, options = {}) => {
        try {
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    },
    
    // Classify hand
    classifyHand: async (handData) => {
        return await API.request('/classify', {
            method: 'POST',
            body: JSON.stringify(handData)
        });
    },
    
    // Train model
    trainModel: async (trainingData) => {
        return await API.request('/train', {
            method: 'POST',
            body: JSON.stringify(trainingData)
        });
    },
    
    // Get available actions
    getActions: async () => {
        return await API.request('/api/actions');
    },
    
    // Get position descriptions
    getPositions: async () => {
        return await API.request('/api/positions');
    }
};

// Event handlers
const EventHandlers = {
    // Initialize page
    init: () => {
        // Add fade-in animation to cards
        document.querySelectorAll('.card').forEach((card, index) => {
            card.style.animationDelay = `${index * 0.1}s`;
            card.classList.add('fade-in');
        });
        
        // Add hover effects
        document.querySelectorAll('.btn').forEach(btn => {
            btn.classList.add('hover-lift');
        });
        
        // Initialize tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    },
    
    // Handle form submissions
    handleFormSubmit: (formId, handler) => {
        const form = document.getElementById(formId);
        if (form) {
            form.addEventListener('submit', handler);
        }
    },
    
    // Handle button clicks
    handleButtonClick: (buttonId, handler) => {
        const button = document.getElementById(buttonId);
        if (button) {
            button.addEventListener('click', handler);
        }
    }
};

// Analytics tracking
const Analytics = {
    // Track page view
    trackPageView: (page) => {
        console.log(`Page viewed: ${page}`);
        // Add your analytics tracking code here
    },
    
    // Track event
    trackEvent: (event, data) => {
        console.log(`Event tracked: ${event}`, data);
        // Add your analytics tracking code here
    },
    
    // Track classification
    trackClassification: (result) => {
        Analytics.trackEvent('hand_classified', {
            predicted_class: result.predicted_class,
            confidence: result.confidence,
            timestamp: new Date().toISOString()
        });
    },
    
    // Track training
    trackTraining: (data) => {
        Analytics.trackEvent('model_trained', {
            num_hands: data.num_hands,
            accuracy: data.accuracy,
            timestamp: new Date().toISOString()
        });
    }
};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    EventHandlers.init();
    
    // Track page view
    Analytics.trackPageView(window.location.pathname);
    
    // Add smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + Enter to submit forms
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const activeForm = document.querySelector('form:focus-within');
            if (activeForm) {
                const submitButton = activeForm.querySelector('button[type="submit"]');
                if (submitButton) {
                    submitButton.click();
                }
            }
        }
        
        // Escape to close modals
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('.modal.show');
            modals.forEach(modal => {
                const modalInstance = bootstrap.Modal.getInstance(modal);
                if (modalInstance) {
                    modalInstance.hide();
                }
            });
        }
    });
});

// Export for use in other scripts
window.PokerRangeClassifier = {
    Utils,
    ChartUtils,
    FormValidator,
    API,
    EventHandlers,
    Analytics
};
