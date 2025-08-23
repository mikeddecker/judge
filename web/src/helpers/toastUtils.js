import { useToast } from 'primevue/usetoast'

export function useToastUtils() {
  const toast = useToast()

  function showToastSuccess(summary = 'Success', detail = null) {
    toast.add({ severity: 'success', summary, detail, life: 3000 })
  }

  return { showToastSuccess }
}

