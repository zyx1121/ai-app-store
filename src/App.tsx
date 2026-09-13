import { Navigate, Route, Routes } from 'react-router'

import { AppSidebar } from '@/components/app-sidebar'
import { SidebarInset, SidebarProvider } from '@/components/ui/sidebar'
import Apps from '@/routes/apps'
import Models from '@/routes/models'
import Setup from '@/routes/setup'
import Store from '@/routes/store'

export default function App() {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <Routes>
          <Route path="/" element={<Navigate to="/setup" replace />} />
          <Route path="/setup" element={<Setup />} />
          <Route path="/store" element={<Store />} />
          <Route path="/models" element={<Models />} />
          <Route path="/apps" element={<Apps />} />
        </Routes>
      </SidebarInset>
    </SidebarProvider>
  )
}
