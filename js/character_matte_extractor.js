import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Comfy.CharacterMatteExtractor",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "CharacterMatteExtractor" && nodeData.name !== "DirectoryCharacterMatteExtractor") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const node = this;

            // Helper to update preview
            function updatePreview(hex) {
                node.previewColor = hex;
            }

            // Find the background_color widget
            const bgColorWidget = this.widgets.find(w => w.name === "background_color");
            if (bgColorWidget) {
                // Update preview on load if value exists
                if (bgColorWidget.value) {
                    updatePreview(bgColorWidget.value);
                }

                // Hook callback
                const originalCallback = bgColorWidget.callback;
                bgColorWidget.callback = function (v) {
                    updatePreview(v);
                    if (originalCallback) {
                        return originalCallback.apply(this, arguments);
                    }
                };
            }

            // Create the button widget
            this.addWidget("button", "Pick Color", null, () => {
                if (!window.EyeDropper) {
                    alert("Your browser does not support the EyeDropper API.");
                    return;
                }

                const eyeDropper = new EyeDropper();
                const abortController = new AbortController();

                eyeDropper.open({ signal: abortController.signal })
                    .then((result) => {
                        const hex = result.sRGBHex;

                        if (bgColorWidget) {
                            bgColorWidget.value = hex;
                            if (bgColorWidget.callback) {
                                bgColorWidget.callback(hex);
                            }
                        } else {
                            updatePreview(hex);
                        }

                        app.graph.setDirtyCanvas(true);
                    })
                    .catch((e) => {
                        console.log("EyeDropper closed", e);
                    });
            });

            // Draw color preview
            const onDrawForeground = node.onDrawForeground;
            node.onDrawForeground = function (ctx) {
                if (onDrawForeground) {
                    onDrawForeground.apply(this, arguments);
                }

                if (this.previewColor) {
                    const lastWidget = this.widgets[this.widgets.length - 1];
                    let y = 40;

                    if (lastWidget && lastWidget.last_y) {
                        y = lastWidget.last_y + 30;
                    }

                    const h = 30;
                    const w = this.size[0] - 20;

                    ctx.save();
                    ctx.fillStyle = this.previewColor;
                    ctx.beginPath();
                    ctx.roundRect(10, y, w, h, 4);
                    ctx.fill();
                    ctx.strokeStyle = "#888";
                    ctx.lineWidth = 1;
                    ctx.stroke();
                    ctx.restore();
                }
            };

            // Increase minimum size to accommodate preview
            const currentSize = node.size;
            node.setSize([currentSize[0], currentSize[1] + 40]);

            return r;
        };
    },
});
