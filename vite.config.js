import { defineConfig } from 'vite';
import { resolve } from 'node:path';

const privateDirectories = ['**/.pose-models/**', '**/.pose-local/**', '**/.venv/**'];

export default defineConfig({
  server: {
    watch: { ignored: privateDirectories },
    fs: { deny: [
      '.env', '.env.*', '*.{crt,pem,key,p12,pfx,cer,der}', '.npmrc', '.yarnrc.yml',
      '**/.git/**', ...privateDirectories,
    ] },
  },
  build: { rollupOptions: { input: {
    main: resolve('index.html'),
    material: resolve('material.html'),
    shuttermuse: resolve('shuttermuse.html'),
  } } },
});
