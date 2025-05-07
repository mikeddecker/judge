import './assets/main.css'

import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import PrimeVue from 'primevue/config';
import Aura from '@primevue/themes/aura'

// PrimeVue components
import Button from "primevue/button"
import Card from 'primevue/card';
import Chart from 'primevue/chart';
import Tabs from 'primevue/tabs';
import TabList from 'primevue/tablist';
import Tab from 'primevue/tab';
import TabPanels from 'primevue/tabpanels';
import TabPanel from 'primevue/tabpanel';


const app = createApp(App)

app.use(router)
app.use(PrimeVue,
    {
        theme: {
            preset: Aura
        }
    }
)

app.component('Button', Button);
app.component('Card', Card);
app.component('Chart', Chart);
app.component('Tabs', Tabs);
app.component('TabList', TabList);
app.component('Tab', Tab);
app.component('TabPanels', TabPanels);
app.component('TabPanel', TabPanel);

app.mount('#app')
