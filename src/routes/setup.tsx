import { useEffect, useRef, useState } from 'react'
import { Check, Copy, Cpu, Download, Plug, RefreshCw } from 'lucide-react'

import { Page } from '@/components/page'
import { ErrorState, LoadingState } from '@/components/state'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress, ProgressLabel, ProgressValue } from '@/components/ui/progress'
import { Separator } from '@/components/ui/separator'
import { useAsync } from '@/hooks/use-async'
import * as api from '@/lib/api/api'
import type { ApiToken } from '@/lib/api/api'
import { toCmdError, type CmdError } from '@/lib/api/error'
import * as hardware from '@/lib/api/hardware'
import type { DeviceProfile, Vendor } from '@/lib/api/hardware'
import * as runtime from '@/lib/api/runtime'
import type { Backend, RuntimeInstall } from '@/lib/api/runtime'
import { formatMb } from '@/lib/format'
import { effectiveMemoryMb, usableMemoryMb } from '@/lib/memory'
import { versionLine } from '@/lib/version'

const VENDOR_LABEL: Record<Vendor, string> = {
  nvidia: 'NVIDIA',
  amd: 'AMD',
  intel: 'Intel',
  apple: 'Apple',
  other: 'Other',
  none: 'None',
}

const BACKEND_LABEL: Record<Backend, string> = {
  cuda: 'CUDA',
  vulkan: 'Vulkan',
  openVino: 'OpenVINO',
  cpu: 'CPU',
}

const ARTIFACT: Record<Backend, string> = {
  cuda: 'win-cuda-13.3-x64',
  vulkan: 'win-vulkan-x64',
  openVino: 'win-openvino-2026.3.1-x64',
  cpu: 'win-cpu-x64',
}

interface TableRow {
  hardware: string
  note: string
}

/** The row of the PLAN section 2.1 table this device matched. */
function tableRow(vendor: Vendor, backend: Backend): TableRow {
  if (backend === 'openVino') {
    return {
      hardware: 'Intel NPU (opt-in)',
      note: 'Experimental: stateless only, single sequence, Q4_0 centric, manual context cap',
    }
  }
  switch (vendor) {
    case 'nvidia':
      return { hardware: 'NVIDIA GPU', note: 'Fastest path, no surprises' }
    case 'amd':
      return {
        hardware: 'AMD GPU / Strix Halo',
        note: 'Decode 13 to 25% faster than HIP on gfx1151; HIP has no official Windows artifact',
      }
    case 'intel':
      return {
        hardware: 'Intel Arc / Panther Lake',
        note: 'SYCL artifact exists and is faster on large models but has open TDR crash issues',
      }
    default:
      return { hardware: 'No usable GPU', note: 'Fallback, store marks large models incompatible' }
  }
}

const STAGES = [
  'Resolving the ggml-org release',
  'Downloading the artifact',
  'Verifying SHA-256 against the release',
  'Unpacking into the runtime folder',
]

interface InstallState {
  status: 'idle' | 'running' | 'done' | 'error'
  stage: string
  percent: number
  install: RuntimeInstall | null
  error: CmdError | null
}

const IDLE: InstallState = { status: 'idle', stage: '', percent: 0, install: null, error: null }

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 py-1.5 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="min-w-0 text-right break-words">{children}</span>
    </div>
  )
}

interface SetupData {
  profile: DeviceProfile
  backend: Backend
}

/** Install flow, PLAN section 7: detect, pick a backend, install the runtime. */
export default function Setup() {
  const [optInNpu, setOptInNpu] = useState(false)
  const [install, setInstall] = useState<InstallState>(IDLE)
  const timer = useRef<number | null>(null)

  const setup = useAsync<SetupData>(async () => {
    const profile = await hardware.detect()
    const backend = await runtime.select(profile, optInNpu)
    return { profile, backend }
  }, [optInNpu])

  useEffect(() => {
    return () => {
      if (timer.current !== null) window.clearInterval(timer.current)
    }
  }, [])

  async function runInstall(backend: Backend) {
    setInstall({ status: 'running', stage: STAGES[0], percent: 4, install: null, error: null })
    timer.current = window.setInterval(() => {
      setInstall((current) => {
        if (current.status !== 'running') return current
        const percent = Math.min(92, current.percent + 4)
        const stage = STAGES[Math.min(STAGES.length - 1, Math.floor(percent / 25))]
        return { ...current, percent, stage }
      })
    }, 120)

    try {
      const result = await runtime.install(backend)
      setInstall({
        status: 'done',
        stage: 'Installed',
        percent: 100,
        install: result,
        error: null,
      })
    } catch (error: unknown) {
      setInstall({
        status: 'error',
        stage: 'Failed',
        percent: 0,
        install: null,
        error: toCmdError(error),
      })
    } finally {
      if (timer.current !== null) window.clearInterval(timer.current)
      timer.current = null
    }
  }

  function reselect() {
    setInstall(IDLE)
    setup.reload()
  }

  return (
    <Page
      title="Setup"
      description="Detect the hardware, show the backend the section 2.1 table picks and install llama-server."
      actions={
        <Button variant="outline" size="sm" onClick={reselect}>
          <RefreshCw />
          Re-select
        </Button>
      }
    >
      {setup.state.status === 'loading' ? <LoadingState label="Detecting the device" /> : null}

      {setup.state.status === 'error' ? (
        <ErrorState error={setup.state.error} what="Hardware detection" onRetry={setup.reload} />
      ) : null}

      {setup.state.status === 'ready' ? (
        <SetupBody
          data={setup.state.data}
          install={install}
          optInNpu={optInNpu}
          onOptInNpu={setOptInNpu}
          onInstall={runInstall}
        />
      ) : null}

      <p className="text-muted-foreground mt-auto pt-4 font-mono text-xs">{versionLine}</p>
    </Page>
  )
}

function SetupBody({
  data,
  install,
  optInNpu,
  onOptInNpu,
  onInstall,
}: {
  data: SetupData
  install: InstallState
  optInNpu: boolean
  onOptInNpu: (value: boolean) => void
  onInstall: (backend: Backend) => void
}) {
  const { profile, backend } = data
  const row = tableRow(profile.gpuVendor, backend)
  const budget = effectiveMemoryMb(profile)

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="size-4" />
            Device
          </CardTitle>
          <CardDescription>What detection read off this machine.</CardDescription>
        </CardHeader>
        <CardContent className="divide-y">
          <Row label="Vendor">{VENDOR_LABEL[profile.gpuVendor]}</Row>
          <Row label="GPU">{profile.gpuName}</Row>
          <Row label="Memory model">
            {profile.memoryModel === 'dedicated' ? 'Dedicated VRAM' : 'Unified memory'}
          </Row>
          <Row label="Effective memory">
            {formatMb(budget)}
            <span className="text-muted-foreground ml-2 text-xs">
              {usableMemoryMb(profile)} MB usable at 90%
            </span>
          </Row>
          <Row label="System RAM">{formatMb(profile.totalRamMb)}</Row>
          <Row label="NPU">{profile.hasNpu ? 'Present' : 'Absent'}</Row>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex flex-wrap items-center gap-2">
            Backend
            <Badge variant="secondary">{BACKEND_LABEL[backend]}</Badge>
          </CardTitle>
          <CardDescription>
            Picked by the fixed table in PLAN section 2.1, never by a probe.
          </CardDescription>
        </CardHeader>
        <CardContent className="divide-y">
          <Row label="Hardware row">{row.hardware}</Row>
          <Row label="Release artifact">
            <span className="font-mono text-xs">{ARTIFACT[backend]}</span>
          </Row>
          <Row label="Reason">{row.note}</Row>
          {profile.hasNpu ? (
            <div className="flex flex-wrap items-center justify-between gap-2 py-2">
              <span className="text-muted-foreground text-sm">Intel NPU opt-in</span>
              <Button
                variant={optInNpu ? 'default' : 'outline'}
                size="sm"
                onClick={() => onOptInNpu(!optInNpu)}
              >
                {optInNpu ? 'On' : 'Off'}
              </Button>
            </div>
          ) : null}

          <div className="flex flex-col gap-3 pt-3">
            <div className="flex flex-wrap items-center gap-2">
              <Button
                onClick={() => onInstall(backend)}
                disabled={install.status === 'running'}
                size="sm"
              >
                {install.status === 'done' ? <Check /> : <Download />}
                {install.status === 'done' ? 'Reinstall' : 'Install runtime'}
              </Button>
              {install.status === 'done' && install.install !== null ? (
                <span className="text-muted-foreground text-xs">
                  tag {install.install.tag}
                </span>
              ) : null}
            </div>

            {install.status === 'running' || install.status === 'done' ? (
              <Progress value={install.percent}>
                <ProgressLabel className="text-xs">{install.stage}</ProgressLabel>
                <ProgressValue className="text-xs" />
              </Progress>
            ) : null}

            {install.status === 'done' && install.install !== null ? (
              <>
                <Separator />
                <p className="text-muted-foreground font-mono text-xs break-all">
                  {install.install.dir}
                </p>
              </>
            ) : null}

            {install.status === 'error' && install.error !== null ? (
              <ErrorState error={install.error} what="Runtime install" />
            ) : null}
          </div>
        </CardContent>
      </Card>

      <LocalApi />
    </div>
  )
}

/**
 * Where an agent reaches this running app.
 *
 * The URL is on the page because it is public; the token is behind a button
 * because it is not. A developer pastes it into the MCP client configuration
 * `aias mcp` reads, and nothing else on this machine needs to see it.
 */
function LocalApi() {
  const access = useAsync<ApiToken>(() => api.token(), [])
  const [copied, setCopied] = useState(false)

  async function copy(token: string) {
    await navigator.clipboard.writeText(token)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1500)
  }

  return (
    <Card className="lg:col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Plug className="size-4" />
          Local API
        </CardTitle>
        <CardDescription>
          The loopback API of PLAN section 6.1. `aias mcp` forwards every agent call to it with
          this token, so the GUI, the CLI and the agent see the same state.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {access.state.status === 'error' ? (
          <ErrorState error={access.state.error} what="Local API" onRetry={access.reload} />
        ) : (
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="font-mono text-xs break-all">
              {access.state.status === 'ready' ? access.state.data.url : 'Starting'}
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={access.state.status !== 'ready'}
              onClick={() => {
                if (access.state.status === 'ready') void copy(access.state.data.token)
              }}
            >
              {copied ? <Check /> : <Copy />}
              {copied ? 'Copied' : 'Copy token'}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
