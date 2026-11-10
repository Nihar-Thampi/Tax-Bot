import { createTheme, type CSSVariablesResolver, type MantineColorsTuple } from '@mantine/core'

// Statute green: the bottle-green of Gazette headings and the SA revenue world.
const ledger: MantineColorsTuple = [
  '#ebf5f1',
  '#d3e9e0',
  '#a6d1c0',
  '#76b89e',
  '#4e9f80',
  '#358e6e',
  '#1f7a5c',
  '#135c45',
  '#0e4a37',
  '#08342a',
]

// Green-cast neutrals for dark mode, matching the paper ground.
const dark: MantineColorsTuple = [
  '#d9e0dc',
  '#b4bdb8',
  '#8d9792',
  '#69736e',
  '#47504b',
  '#343c38',
  '#262d2a',
  '#1b211e',
  '#151a18',
  '#0f1311',
]

export const theme = createTheme({
  primaryColor: 'ledger',
  primaryShade: { light: 7, dark: 5 },
  colors: { ledger, dark },
  // Public Sans for the interface (a government-document face); Source Serif 4 is kept for
  // the statute's own words, see --app-font-statute.
  fontFamily: "'Public Sans', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif",
  headings: {
    fontFamily: "'Public Sans', ui-sans-serif, system-ui, sans-serif",
    fontWeight: '650',
  },
  defaultRadius: 'md',
  radius: { xs: '4px', sm: '6px', md: '10px', lg: '14px', xl: '20px' },
  cursorType: 'pointer',
  focusRing: 'auto',
  components: {
    Button: { defaultProps: { fw: 500 } },
    Modal: { defaultProps: { radius: 'lg', overlayProps: { backgroundOpacity: 0.45, blur: 2 } } },
    Menu: { defaultProps: { radius: 'md', shadow: 'md' } },
  },
})

export const cssVariablesResolver: CSSVariablesResolver = () => ({
  variables: {
    '--app-font-statute': "'Source Serif 4', ui-serif, Georgia, serif",
  },
  light: {
    '--mantine-color-body': '#f3f5f4',
    '--mantine-color-text': '#17201c',
    '--app-surface': '#ffffff',
    '--app-border': '#d9e0dc',
    '--app-rule': '#c9d3ce',
    '--app-muted-surface': '#e9eeeb',
  },
  dark: {
    '--mantine-color-body': '#121715',
    '--app-surface': '#1a201d',
    '--app-border': '#29312d',
    '--app-rule': '#36403b',
    '--app-muted-surface': '#171d1a',
  },
})
