<template>
	<div class="openapi-view">
		<div v-if="loading" class="loading">Loading API docs…</div>
		<div v-if="error" class="error">
			<strong>Error:</strong> {{ error }}
			<div><a :href="docsUrl" target="_blank" rel="noopener">Open `/docs` in new tab</a></div>
		</div>
		<div id="swagger-ui" v-show="!error" style="height:100%;"></div>
	</div>
</template>

<script>
export default {
	name: 'OpenApiView',
	data() {
		const base = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')
		return {
			loading: true,
			error: null,
			docsUrl: base ? `${base}/docs` : '/api/docs',
			openapiUrl: base ? `${base}/openapi.json` : '/api/openapi.json',
			_abort: null,
		}
	},
	methods: {
		_appendOnce(tag, attrs) {
			// tag: 'link' or 'script', attrs: {src/href, ...}
			const selector = Object.entries(attrs).map(([k, v]) => `[${k}="${v}"]`).join('')
			if (document.querySelector(`${tag}${selector}`)) return false
			const el = document.createElement(tag)
			for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v)
			document.head.appendChild(el)
			return true
		},
		_initSwagger() {
			try {
				const bundle = window.SwaggerUIBundle
				if (!bundle) throw new Error('SwaggerUIBundle not available')

				const standalone = (bundle && bundle.SwaggerUIStandalonePreset) || window.SwaggerUIStandalonePreset
				const presets = [bundle.presets.apis]
				if (standalone) presets.push(standalone)

				bundle({
					url: this.openapiUrl,
					dom_id: '#swagger-ui',
					presets,
					layout: standalone ? 'StandaloneLayout' : 'BaseLayout',
					deepLinking: true,
				})
				this.loading = false
			} catch (err) {
				this.error = String(err)
				this.loading = false
			}
		},
		_ensureSwaggerLoaded() {
			const src = 'https://unpkg.com/swagger-ui-dist@4/swagger-ui-bundle.js'
			// If bundle already present, init
			if (window.SwaggerUIBundle) return this._initSwagger()

			// If script exists but bundle not ready, attach listener
			const existing = document.querySelector(`script[src="${src}"]`)
			if (existing) {
				existing.addEventListener('load', this._initSwagger, { once: true })
				existing.addEventListener('error', () => { this.error = 'Failed to load Swagger UI script' })
				return
			}

			// Add stylesheet and script
			this._appendOnce('link', { rel: 'stylesheet', href: 'https://unpkg.com/swagger-ui-dist@4/swagger-ui.css', 'data-swagger': '1' })
			this._appendOnce('script', { src, 'data-swagger': '1' })
			const added = document.querySelector(`script[src="${src}"]`)
			if (added) {
				added.addEventListener('load', this._initSwagger, { once: true })
				added.addEventListener('error', () => { this.error = 'Failed to load Swagger UI script' })
			}
		}
	},
	mounted() {
		// Try a quick fetch to provide early feedback; but always attempt to load the UI
		this._abort = new AbortController()
		const timeout = setTimeout(() => this._abort.abort(), 4000)
		fetch(this.openapiUrl, { credentials: 'include', signal: this._abort.signal })
			.then(res => {
				clearTimeout(timeout)
				if (!res.ok) {
					// still try to load Swagger UI; it may work depending on CORS/auth
					this._ensureSwaggerLoaded()
					this.error = `OpenAPI returned ${res.status}`
					this.loading = false
					return
				}
				this._ensureSwaggerLoaded()
			})
			.catch(() => {
				// network/CORS issue: still attempt to load UI (it will fetch the spec in-browser)
				this._ensureSwaggerLoaded()
			})
	},
	beforeUnmount() {
		if (this._abort) this._abort.abort()
	}
}
</script>

<style scoped>
.openapi-view { height: 100%; display: flex; flex-direction: column; }
.loading { padding: 1rem; color: var(--text-color, #333); }
.error { padding: 1rem; color: var(--error-color, #b00020); }
#swagger-ui { flex: 1 1 auto; min-height: 0; }
</style>

