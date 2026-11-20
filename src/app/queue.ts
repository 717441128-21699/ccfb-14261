/** 批量处理队列:顺序执行,报告进度 */

export interface QueueItem {
  id: number;
  name: string;
  file: File;
  microscope: string;
  magnification: string;
  status: '等待' | '处理中' | '完成' | '失败';
  progress: number; // 0..1
  detail: string;
  error?: string;
}

export type QueueListener = (items: QueueItem[]) => void;

export class ProcessQueue {
  private items: QueueItem[] = [];
  private nextId = 1;
  private running = false;
  private listeners: QueueListener[] = [];

  onChange(fn: QueueListener): void {
    this.listeners.push(fn);
  }

  private emit(): void {
    for (const fn of this.listeners) fn(this.items);
  }

  add(file: File, microscope: string, magnification: string): void {
    this.items.push({
      id: this.nextId++,
      name: file.name,
      file,
      microscope,
      magnification,
      status: '等待',
      progress: 0,
      detail: '',
    });
    this.emit();
  }

  clear(): void {
    this.items = this.items.filter((i) => i.status === '处理中');
    this.emit();
  }

  get pending(): QueueItem[] {
    return this.items.filter((i) => i.status === '等待');
  }

  update(id: number, patch: Partial<QueueItem>): void {
    const it = this.items.find((i) => i.id === id);
    if (it) Object.assign(it, patch);
    this.emit();
  }

  /** 顺序处理队列中所有「等待」项 */
  async run(processor: (item: QueueItem, onProgress: (p: number, detail: string) => void) => Promise<void>): Promise<void> {
    if (this.running) return;
    this.running = true;
    try {
      for (const item of this.items) {
        if (item.status !== '等待') continue;
        this.update(item.id, { status: '处理中', progress: 0 });
        try {
          await processor(item, (p, detail) => this.update(item.id, { progress: p, detail }));
          this.update(item.id, { status: '完成', progress: 1, detail: '已导出' });
        } catch (e) {
          this.update(item.id, { status: '失败', error: String(e), detail: '' });
        }
      }
    } finally {
      this.running = false;
    }
  }
}
