import { Boxes, Cpu, HardDrive, Store } from 'lucide-react'
import { NavLink } from 'react-router'

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from '@/components/ui/sidebar'
import { isMock } from '@/lib/api/invoke'
import { versionLine } from '@/lib/version'

const NAV = [
  { to: '/setup', label: 'Setup', hint: 'Device, backend, install', icon: Cpu },
  { to: '/store', label: 'Store', hint: 'Models and apps to add', icon: Store },
  { to: '/models', label: 'Models', hint: 'Weights and instances', icon: HardDrive },
  { to: '/apps', label: 'Apps', hint: 'Subscribed apps and logs', icon: Boxes },
]

export function AppSidebar() {
  return (
    <Sidebar>
      <SidebarHeader>
        <div className="px-2 py-1.5">
          <p className="text-sm font-medium">AI App Store</p>
          <p className="text-muted-foreground text-xs">One runtime contract for local AI apps</p>
        </div>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Platform</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {NAV.map((item) => (
                <SidebarMenuItem key={item.to}>
                  <NavLink to={item.to}>
                    {({ isActive }) => (
                      <SidebarMenuButton render={<span />} isActive={isActive} className="h-auto py-2">
                        <item.icon />
                        <span className="flex min-w-0 flex-col">
                          <span className="truncate text-sm leading-5">{item.label}</span>
                          <span className="text-muted-foreground truncate text-xs leading-4">
                            {item.hint}
                          </span>
                        </span>
                      </SidebarMenuButton>
                    )}
                  </NavLink>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      {isMock ? (
        <SidebarFooter>
          <p className="text-muted-foreground px-2 pb-1 font-mono text-xs">
            {versionLine} mock data
          </p>
        </SidebarFooter>
      ) : null}
    </Sidebar>
  )
}
