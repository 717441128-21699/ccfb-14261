/** PSF 库:IndexedDB 持久化,按显微镜 + 倍率建档 */

export interface PsfRecord {
  id?: number;
  microscope: string;
  magnification: string;
  size: number;
  data: Float32Array;
  createdAt: number;
  notes?: string;
}

const DB_NAME = 'micro-deconv';
const STORE = 'psfs';

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function savePsf(rec: PsfRecord): Promise<number> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    const req = tx.objectStore(STORE).add({ ...rec, data: rec.data.buffer.slice(0) });
    req.onsuccess = () => resolve(req.result as number);
    req.onerror = () => reject(req.error);
  });
}

export async function listPsfs(): Promise<PsfRecord[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const req = db.transaction(STORE, 'readonly').objectStore(STORE).getAll();
    req.onsuccess = () => {
      const rows = req.result as Array<Omit<PsfRecord, 'data'> & { data: ArrayBuffer }>;
      resolve(rows.map((r) => ({ ...r, data: new Float32Array(r.data) })));
    };
    req.onerror = () => reject(req.error);
  });
}

export async function deletePsf(id: number): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const req = db.transaction(STORE, 'readwrite').objectStore(STORE).delete(id);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

/** 按显微镜+倍率精确匹配 PSF */
export async function findPsf(microscope: string, magnification: string): Promise<PsfRecord | undefined> {
  const all = await listPsfs();
  return all.find((r) => r.microscope === microscope && r.magnification === magnification);
}
