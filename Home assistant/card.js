class VideoStreamCard extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
    }

    setConfig(config) {
        if (!config.entity) {
            throw new Error('Please define an entity');
        }
        this.config = config;
    }

    set hass(hass) {
        const content = document.createElement('div');
        content.innerHTML = `
            <ha-card header="${this.config.title || 'Video Stream'}">
                <video 
                    autoplay 
                    controls
                    style="width: 100%; height: auto;"
                    src="${this.config.stream_url}">
                </video>
            </ha-card>
        `;

        if (!this.shadowRoot.hasChildNodes()) {
            this.shadowRoot.appendChild(content);
        }
    }

    static getStubConfig() {
        return {
            entity: "",
            title: "Video Stream",
            stream_url: ""
        };
    }
}

customElements.define('video-stream-card', VideoStreamCard);

window.customCards = window.customCards || [];
window.customCards.push({
    type: "video-stream-card",
    name: "Video Stream Card",
    preview: false
});
