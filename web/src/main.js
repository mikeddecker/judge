import './assets/main.css'

import { createApp } from 'vue'
import { createPinia } from "pinia";
import App from './App.vue'
import router from './router'
import PrimeVue from 'primevue/config';
import Aura from '@primevue/themes/aura'
import ConfirmationService from 'primevue/confirmationservice';

// PrimeVue components
import Button from "primevue/button"
import Card from 'primevue/card';
import Chart from 'primevue/chart';
import Column from 'primevue/column';
import ConfirmPopup from 'primevue/confirmpopup';
import DataTable from 'primevue/datatable';
import Divider from 'primevue/divider';
import InputNumber from 'primevue/inputnumber';
import InputText from 'primevue/inputtext';
import { IftaLabel } from 'primevue';
import Listbox from 'primevue/listbox';
import RadioButton from 'primevue/radiobutton';
import Select from 'primevue/select';
import Tabs from 'primevue/tabs';
import TabList from 'primevue/tablist';
import Tab from 'primevue/tab';
import TabPanels from 'primevue/tabpanels';
import TabPanel from 'primevue/tabpanel';
import { useSkillStore } from './stores/skillStore';

const app = createApp(App)

app.use(router)
app.use(PrimeVue,
    {
        theme: {
            preset: Aura,
            options: {
                cssLayer: {
                    name: 'primevue',
                    order: 'theme, base, primevue'
                }
            }
        }
    }
)
app.use(createPinia());
app.use(ConfirmationService);

app.component('Button', Button);
app.component('Card', Card);
app.component('Chart', Chart);
app.component('Column', Column);
app.component('ConfirmPopup', ConfirmPopup);
app.component('DataTable', DataTable);
app.component('Divider', Divider);
app.component('InputNumber', InputNumber)
app.component('InputText', InputText)
app.component('IftaLabel', IftaLabel)
app.component('Listbox', Listbox)
app.component('RadioButton', RadioButton);
app.component('Select', Select);
app.component('Tabs', Tabs);
app.component('TabList', TabList);
app.component('Tab', Tab);
app.component('TabPanels', TabPanels);
app.component('TabPanel', TabPanel);

app.mount('#app')

// Load all data
const skillStore = useSkillStore().loadData()

