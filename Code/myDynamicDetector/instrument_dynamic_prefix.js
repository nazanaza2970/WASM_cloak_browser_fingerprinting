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
            console.warn(`Cannot hook non-function: ${prefix || methodName}`);
            return;
        }

        // Try to determine prefix dynamically first
        const dynamicPrefix =
            Object.entries(window).find(([key, value]) => value?.prototype === prototype)?.[0] ||
            prototype.constructor?.name ||
            null;

        // Use dynamic prefix if found; otherwise, fallback to provided prefix or default
        const finalPrefix = dynamicPrefix || prefix || 'UnknownPrototype';

        prototype[methodName] = function (...args) {
            const returnValue = originalMethod.apply(this, args);
            const symbol = `${finalPrefix}.${methodName.toLowerCase()}`;
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
            console.warn(`Cannot find property descriptor for: ${prefix || propertyName}`);
            return;
        }

        // Try to determine prefix dynamically first
        const dynamicPrefix =
            Object.entries(window).find(([key, value]) => value?.prototype === prototype)?.[0] ||
            prototype.constructor?.name ||
            null;

        // Use dynamic prefix if found; otherwise, fallback to provided prefix or default
        const finalPrefix = dynamicPrefix || prefix || 'UnknownPrototype';

        Object.defineProperty(prototype, propertyName, {
            get: function() {
                if (!descriptor.get) return undefined;
                const originalValue = descriptor.get.apply(this);
                const symbol = `${finalPrefix}.${propertyName.toLowerCase()}`;
                logAPICall(symbol, {}, originalValue); // Getter has no args
                return originalValue;
            },
            set: function(newValue) {
                if (descriptor.set) {
                    descriptor.set.apply(this, [newValue]);
                }
                const symbol = `${finalPrefix}.${propertyName.toLowerCase()}`;
                logAPICall(symbol, [newValue], undefined); // Setter logs assigned value
            },
            configurable: true
        });
    }


    //////////////////////////////////////////////
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

    // --- Hooking navigator.plugins (a property) --- cop
    // const pluginsDescriptor = Object.getOwnPropertyDescriptor(Navigator.prototype, 'plugins');
    // if (pluginsDescriptor) {
    //     Object.defineProperty(Navigator.prototype, 'plugins', {
    //         get: function() {
    //             const originalValue = pluginsDescriptor.get.apply(this);
    //             logAPICall('navigator.plugins', {}, originalValue);
    //             return originalValue;
    //         },
    //         configurable: true
    //     });
    // }
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

    ///////////////////////////////////////
    // instrument_dom.js
    console.log("Applying DOM hooks...");

    // Hook document.cookie
    try {
        const cookieDescriptor = Object.getOwnPropertyDescriptor(Document.prototype, 'cookie');
        if(cookieDescriptor) {
            Object.defineProperty(document, 'cookie', {
                get: function() {
                    const val = cookieDescriptor.get.apply(document);
                    logAPICall('window.document.cookie', {}, val);
                    return val;
                },
                set: function(val) {
                    logAPICall('window.document.cookie', {set: val}, '');
                    cookieDescriptor.set.apply(document, [val]);
                },
                configurable: true
            });
        }
    } catch (e) { console.error('Failed to hook document.cookie', e); }
    // For methods on Document
    hookMethod(Document.prototype, 'getElementsByTagName', 'document');

    // For methods on HTMLCanvasElement
    hookMethod(HTMLCanvasElement.prototype, 'getElementsByTagName', 'htmlcanvaselement');
    hookMethod(HTMLCanvasElement.prototype, 'toDataURL', 'htmlcanvaselement');
    hookMethod(HTMLCanvasElement.prototype, 'toBlob', 'htmlcanvaselement');
    hookMethod(HTMLCanvasElement.prototype, 'getContext', 'htmlcanvaselement');
    hookMethod(HTMLCanvasElement.prototype, 'addEventListener', 'htmlcanvaselement');
    hookProperty(HTMLCanvasElement.prototype, 'width', 'htmlcanvaselement');
    hookProperty(HTMLCanvasElement.prototype, 'height', 'htmlcanvaselement');

    ///////////////////////////////////////
    /**
     * A generic function to hook only the setter of a property.
     * Useful for event handlers like 'oncomplete'.
     * @param {object} prototype - The prototype object.
     * @param {string} propertyName - The name of the property to hook.
     * @param {string} prefix - The prefix for the symbol name.
     */
    function hookSetter(prototype, propertyName, prefix) {
        const descriptor = Object.getOwnPropertyDescriptor(prototype, propertyName);
        if (!descriptor) {
            console.warn(`Cannot find property descriptor for: ${prefix || propertyName}`);
            return;
        }

        // Try to determine prefix dynamically first
        const dynamicPrefix =
            Object.entries(window).find(([key, value]) => value?.prototype === prototype)?.[0] ||
            prototype.constructor?.name ||
            null;

        // Use dynamic prefix if found; otherwise, fallback to provided prefix or default
        const finalPrefix = dynamicPrefix || prefix || 'UnknownPrototype';

        Object.defineProperty(prototype, propertyName, {
            get: descriptor.get, // Preserve original getter
            set: function (newValue) {
                // Apply the original setter if it exists
                if (descriptor.set) {
                    descriptor.set.apply(this, [newValue]);
                }
                const symbol = `${finalPrefix}.${propertyName.toLowerCase()}`;
                logAPICall(symbol, [newValue], undefined); // Log the value being set
            },
            configurable: true
        });
    }
    // --- Hooking Web Audio APIs ---
    console.log("Applying Web Audio hooks...");

    // For AudioContext
    if (window.AudioContext) {
        const prefix = 'audiocontext';
        hookMethod(AudioContext.prototype, 'createChannelMerger', prefix);
        hookMethod(AudioContext.prototype, 'createBuffer', prefix);
        hookMethod(AudioContext.prototype, 'createScriptProcessor', prefix);
    }

    // For OfflineAudioContext
    if (window.OfflineAudioContext) {
        const prefix = 'offlineaudiocontext';
        hookProperty(OfflineAudioContext.prototype, 'destination', prefix);
        hookSetter(OfflineAudioContext.prototype, 'oncomplete', prefix);
        hookMethod(OfflineAudioContext.prototype, 'createScriptProcessor', prefix);
        hookMethod(OfflineAudioContext.prototype, 'createOscillator', prefix);
        hookMethod(OfflineAudioContext.prototype, 'createDynamicsCompressor', prefix);
        hookMethod(OfflineAudioContext.prototype, 'startRendering', prefix);
    }

    ///////////////////////////////////////
    // instrument_webgl.js
    function hookWebGL(proto, prefix) { // <-- Add prefix here
        if (!proto) return;
        hookMethod(proto, 'viewport', prefix);
        hookMethod(proto, 'scissor', prefix);
        hookMethod(proto, 'readPixels', prefix);
        hookMethod(proto, 'bufferData', prefix);
        hookMethod(proto, 'getAttribLocation', prefix);
        hookMethod(proto, 'getUniformLocation', prefix);
        hookMethod(proto, 'getShaderParameter', prefix);
        hookMethod(proto, 'getProgramParameter', prefix);
        hookMethod(proto, 'getActiveAttrib', prefix);
        hookMethod(proto, 'getActiveUniform', prefix);
        hookMethod(proto, 'bindAttribLocation', prefix);
        hookMethod(proto, 'uniform2f', prefix);
        hookMethod(proto, 'vertexAttribPointer', prefix);
        hookMethod(proto, 'drawArrays', prefix);
        hookMethod(proto, 'clearColor', prefix);
        hookMethod(proto, 'getShaderPrecisionFormat', prefix);
        hookMethod(proto, 'texParameteri', prefix);
        hookMethod(proto, 'pixelStorei', prefix);
        hookMethod(proto, 'drawElements', prefix);
        hookMethod(proto, 'blendFunc', prefix);
        hookMethod(proto, 'framebufferTexture2D', prefix);
        hookMethod(proto, 'texImage2D', prefix);
        hookMethod(proto, 'blendColor', prefix);
        hookMethod(proto, 'colorMask', prefix);
        hookMethod(proto, 'blendFuncSeparate', prefix);
        hookMethod(proto, 'framebufferRenderbuffer', prefix);
        hookMethod(proto, 'blendEquationSeparate', prefix);
        hookMethod(proto, 'hint', prefix);
        hookMethod(proto, 'stencilOp', prefix);
        hookMethod(proto, 'renderbufferStorage', prefix);
    }
    // --- Hooking WebGL APIs ---
    console.log("Applying WebGL hooks...");

    try {
        // Hook WebGL1 methods
        hookWebGL(
            window.WebGLRenderingContext && WebGLRenderingContext.prototype,
            'webglrenderingcontext' // <-- Prefix for WebGL1
        );

        // Hook WebGL2 methods
        hookWebGL(
            window.WebGL2RenderingContext && WebGL2RenderingContext.prototype,
            'webgl2renderingcontext' // <-- Prefix for WebGL2
        );
    } catch (e) {
        console.error("Could not hook WebGL methods.", e);
    }

    ///////////////////////////////////////
    // --- Hooking WebRTC APIs ---
    console.log("Applying WebRTC hooks...");

    if (window.RTCPeerConnection) {
        const prefix = 'rtcpeerconnection'; // <-- Define the prefix

        hookProperty(RTCPeerConnection.prototype, 'localDescription', prefix);
        hookSetter(RTCPeerConnection.prototype, 'onicecandidate', prefix);
        hookMethod(RTCPeerConnection.prototype, 'createDataChannel', prefix);
        hookMethod(RTCPeerConnection.prototype, 'createOffer', prefix);
    }


    console.log("Instrumentation script injected and active.");
})();