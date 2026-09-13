import { Badge } from '@/components/ui/badge'
import type { Fit } from '@/lib/api/models'

const LABEL: Record<Fit, string> = {
  ready: 'Ready',
  maybe: 'Maybe',
  incompatible: 'Incompatible',
}

const VARIANT: Record<Fit, 'default' | 'secondary' | 'outline'> = {
  ready: 'default',
  maybe: 'secondary',
  incompatible: 'outline',
}

/**
 * The badge of PLAN section 2.2, computed by `models_fit` against this device.
 *
 * Ready means the context the planner would hand out is still a usable one, at
 * least 4096 tokens over one slot, under the 90% ceiling. Maybe means it only
 * fits at the 2048 token floor, Incompatible means it does not fit at all.
 */
export function FitBadge({ fit, className }: { fit: Fit; className?: string }) {
  return (
    <Badge variant={VARIANT[fit]} className={className}>
      {LABEL[fit]}
    </Badge>
  )
}
