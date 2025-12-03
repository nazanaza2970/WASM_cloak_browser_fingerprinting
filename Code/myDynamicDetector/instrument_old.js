// This script is injected by Playwright before the page's own scripts run.

(function() {
    'use strict';

    // Create a global array to store API call logs if it doesn't exist.
    window.__API_LOGS__ = window.__API_LOGS__ || [];

    /**
     * A helper function to log API calls.
     * @param {string} symbol - The full API symbol (e.g., 'navigator.useragent').
     * @param {object} args - The arguments passed to the function.
     * @param {*} value - The return value of the call.
     */
    function logAPICall(symbol, args, value) {
        // To avoid logging storms from getters, check if the last log is identical.
        const lastLog = window.__API_LOGS__[window.__API_LOGS__.length - 1];
        if (lastLog && lastLog.symbol === symbol && lastLog.value === value) {
            return;
        }
        window.__API_LOGS__.push({
            symbol: symbol,
            argument: JSON.stringify(args), // Stringify to handle complex objects
            value: value,
            timestamp: performance.now()
        });
    }

    /**
     * A generic function to hook a method on a given prototype.
     * @param {object} prototype - The prototype object (e.g., CanvasRenderingContext2D.prototype).
     * @param {string} methodName - The name of the method to hook.
     * @param {string} prefix - The prefix for the symbol name (e.g., 'canvasrenderingcontext2d').
     */
    function hookMethod(prototype, methodName, prefix) {
        if (!(methodName in prototype)) return; // Failsafe
        const originalMethod = prototype[methodName];
        if (typeof originalMethod !== 'function') {
            console.warn(`Cannot hook non-function: ${prefix}.${methodName}`);
            return;
        }

        prototype[methodName] = function(...args) {
            const returnValue = originalMethod.apply(this, args);
            const symbol = `${prefix}.${methodName.toLowerCase()}`;
            logAPICall(symbol, args, returnValue);
            return returnValue;
        };
    }

    /**
     * A generic function to hook a property (both get and set) on a given prototype.
     * @param {object} prototype - The prototype object.
     * @param {string} propertyName - The name of the property to hook.
     * @param {string} prefix - The prefix for the symbol name.
     */
    function hookProperty(prototype, propertyName, prefix) {
        const descriptor = Object.getOwnPropertyDescriptor(prototype, propertyName);
        if (!descriptor) {
            console.warn(`Cannot find property descriptor for: ${prefix}.${propertyName}`);
            return;
        }

        Object.defineProperty(prototype, propertyName, {
            get: function() {
                const originalValue = descriptor.get.apply(this);
                const symbol = `${prefix}.${propertyName.toLowerCase()}`;
                logAPICall(symbol, {}, originalValue); // Getter has no args
                return originalValue;
            },
            set: function(newValue) {
                if (descriptor.set) {
                    descriptor.set.apply(this, [newValue]);
                }
                const symbol = `${prefix}.${propertyName.toLowerCase()}`;
                // Setter's argument is the value being set, no return value to log
                logAPICall(symbol, [newValue], undefined);
            },
            configurable: true
        });
    }


    // --- Hooking navigator.userAgent (a property) ---
    const userAgentDescriptor = Object.getOwnPropertyDescriptor(Navigator.prototype, 'userAgent');
    if (userAgentDescriptor) {
        Object.defineProperty(Navigator.prototype, 'userAgent', {
            get: function() {
                const originalValue = userAgentDescriptor.get.apply(this);
                logAPICall('navigator.useragent', {}, originalValue);
                return originalValue;
            },
            configurable: true
        });
    }


    // --- Hooking Canvas APIs ---
    const canvasProperties = [
        'strokeStyle', 'fillStyle', 'shadowColor', 'filter', 'font'
    ];

    const canvasMethods = [
        'fillRect', 'rect', 'arc', 'getImageData', 'strokeRect', 'clearRect',
        'createRadialGradient', 'bezierCurveTo', 'fillText', 'strokeText',
        'createPattern', 'createLinearGradient', 'putImageData', 'setLineDash',
        'createImageData', 'arcTo', 'moveTo', 'lineTo', 'translate',
        'drawImage', 'quadraticCurveTo', 'save', 'restore', 'addEventListener',
        'measureText'
    ];

    const proto = CanvasRenderingContext2D.prototype;
    const prefix = 'canvasrenderingcontext2d';

    canvasProperties.forEach(prop => hookProperty(proto, prop, prefix));
    canvasMethods.forEach(method => hookMethod(proto, method, prefix));


    console.log("Instrumentation script injected and active.");
})();