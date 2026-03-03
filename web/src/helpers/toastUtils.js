import { useToast } from 'primevue/usetoast'

export function useToastUtils() {
  const toast = useToast()

  function showToastSuccess(summary = 'Success', detail = null) {
    toast.add({ severity: 'success', summary, detail, life: 3000 })
  }

  function showToastError(detail = null, summary = 'Error') {
    toast.add({ severity: 'error', summary, detail, life: 4250 })
  }

  return { showToastSuccess, showToastError }
}

