import './assets/main.css'

import { createApp } from 'vue'
import { createPinia } from "pinia";
import App from './App.vue'
import router from './router'
import PrimeVue from 'primevue/config';
import Aura from '@primevue/themes/aura'
import ConfirmationService from 'primevue/confirmationservice';
import Vue3Shortkey from '@gregdev/vue3-shortkey'

// PrimeVue components
import Button from "primevue/button"
import Card from 'primevue/card';
import Chart from 'primevue/chart';
import Checkbox from 'primevue/checkbox';
import Column from 'primevue/column';
import ConfirmPopup from 'primevue/confirmpopup';
import DataTable from 'primevue/datatable';
import Divider from 'primevue/divider';
import Drawer from 'primevue/drawer';
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
import Tooltip from 'primevue/tooltip'
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
app.use(Vue3Shortkey, {
  prevent: ['input', 'textarea'], // ignore shortcuts while typing
  capture: true,                  // listen during capture phase
  propagation: false,             // stop event propagation by default
})

app.component('Button', Button);
app.component('Card', Card);
app.component('Chart', Chart);
app.component('Checkbox', Checkbox);
app.component('Column', Column);
app.component('ConfirmPopup', ConfirmPopup);
app.component('DataTable', DataTable);
app.component('Divider', Divider);
app.component('Drawer', Drawer);
app.component('InputNumber', InputNumber);
app.component('InputText', InputText);
app.component('IftaLabel', IftaLabel);
app.component('Listbox', Listbox);
app.component('RadioButton', RadioButton);
app.component('Select', Select);
app.component('Tabs', Tabs);
app.component('TabList', TabList);
app.component('Tab', Tab);
app.component('TabPanels', TabPanels);
app.component('TabPanel', TabPanel);

app.directive('tooltip', Tooltip)

app.mount('#app')

// Load all data
const skillStore = useSkillStore().loadData()

