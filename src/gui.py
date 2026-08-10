"""用户端动物识别系统 — 本地GUI演示程序"""
import os, io, sys, time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import models, transforms

CLASSES_ZH = ['猫', '牛', '鸡', '狗', '鸭', '狮子', '羊', '虎']
CLASSES_EN = ['cat', 'cattle', 'chicken', 'dog', 'duck', 'lion', 'sheep', 'tiger']
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', '最终动物识别模型.pth')
UNKNOWN_THRESHOLD = 0.60

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

COLORS = {
    'bg':       '#1a1a2e',
    'panel':    '#16213e',
    'accent':   '#0f3460',
    'primary':  '#1E6FB5',
    'success':  '#1B8A4A',
    'warning':  '#E86A17',
    'danger':   '#D93A3A',
    'text':     '#EAEAEA',
    'text_dim': '#8892A4',
    'white':    '#FFFFFF',
    'bar_grad': ['#1B8A4A', '#3CB371', '#FFD700', '#FF8C00', '#D93A3A'],
}

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def build_model():
    """构建 ResNet18 模型并加载预训练权重"""
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    fc_keys = [k for k in state if k.startswith('fc.')]
    last_w = sorted([k for k in fc_keys if k.endswith('.weight') and 'running' not in k], reverse=True)
    if not last_w:
        raise ValueError('无法识别模型FC层')
    n_cls = state[last_w[0]].shape[0]
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.BatchNorm1d(512), nn.Dropout(0.5),
        nn.Linear(512, 512), nn.ReLU(inplace=True),
        nn.BatchNorm1d(512), nn.Dropout(0.3),
        nn.Linear(512, n_cls)
    )
    m.load_state_dict(state, strict=False)
    m.to(DEVICE).eval()
    return m


class RecognitionApp:
    """动物识别 GUI 应用程序"""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('动物种类智能识别系统 — CNN · ResNet18')
        self.root.geometry('1100x720')
        self.root.configure(bg=COLORS['bg'])
        self.root.minsize(1000, 650)
        try:
            self.root.iconbitmap(default='')
        except:
            pass

        self.model = None
        self.current_image = None
        self.current_tkimg = None
        self.result_data = None

        self._build_ui()
        self._load_model()

    def _build_ui(self):
        # Title bar
        title_frame = tk.Frame(self.root, bg=COLORS['panel'], height=56)
        title_frame.pack(fill=tk.X, side=tk.TOP)
        title_frame.pack_propagate(False)
        tk.Label(title_frame, text='CNN 动物种类智能识别系统', font=('Microsoft YaHei', 18, 'bold'),
                 fg=COLORS['white'], bg=COLORS['panel']).pack(side=tk.LEFT, padx=20, pady=10)
        tk.Label(title_frame, text='ResNet18 · PyTorch', font=('Microsoft YaHei', 11),
                 fg=COLORS['text_dim'], bg=COLORS['panel']).pack(side=tk.LEFT, padx=(0, 0))
        self.status_lbl = tk.Label(title_frame, text='● 模型加载中...', font=('Microsoft YaHei', 11),
                                    fg=COLORS['warning'], bg=COLORS['panel'])
        self.status_lbl.pack(side=tk.RIGHT, padx=20)

        # Main area
        main = tk.Frame(self.root, bg=COLORS['bg'])
        main.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Left: image panel
        left = tk.Frame(main, bg=COLORS['panel'], width=520, height=500)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        left.pack_propagate(False)

        tk.Label(left, text='识别图像', font=('Microsoft YaHei', 13, 'bold'),
                 fg=COLORS['white'], bg=COLORS['panel']).pack(pady=(12, 4))

        self.img_canvas = tk.Canvas(left, bg=COLORS['bg'], highlightthickness=0)
        self.img_canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        self.img_canvas_text = self.img_canvas.create_text(
            260, 220, text='点击下方按钮选择图片\n或拖拽图片到此处',
            font=('Microsoft YaHei', 13), fill=COLORS['text_dim'], justify=tk.CENTER
        )
        self.img_canvas_id = None

        # Button bar
        btn_frame = tk.Frame(left, bg=COLORS['panel'])
        btn_frame.pack(fill=tk.X, padx=12, pady=(0, 12))

        self.select_btn = tk.Button(btn_frame, text='  选择图片  ', font=('Microsoft YaHei', 12),
                                     bg=COLORS['primary'], fg=COLORS['white'], relief=tk.FLAT,
                                     cursor='hand2', command=self._select_image,
                                     activebackground=COLORS['accent'], activeforeground=COLORS['white'])
        self.select_btn.pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(btn_frame, text='  清  除  ', font=('Microsoft YaHei', 12),
                  bg=COLORS['accent'], fg=COLORS['text_dim'], relief=tk.FLAT,
                  cursor='hand2', command=self._clear,
                  activebackground=COLORS['danger'], activeforeground=COLORS['white']
                  ).pack(side=tk.LEFT)

        self.infer_btn = tk.Button(btn_frame, text='  开始识别  ', font=('Microsoft YaHei', 12, 'bold'),
                                    bg=COLORS['success'], fg=COLORS['white'], relief=tk.FLAT,
                                    cursor='hand2', command=self._predict, state=tk.DISABLED,
                                    activebackground='#145A34', activeforeground=COLORS['white'])
        self.infer_btn.pack(side=tk.RIGHT)

        # Right: results panel
        right = tk.Frame(main, bg=COLORS['panel'], width=520)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        right.pack_propagate(False)

        tk.Label(right, text='识别结果', font=('Microsoft YaHei', 13, 'bold'),
                 fg=COLORS['white'], bg=COLORS['panel']).pack(pady=(12, 8))

        self.top_frame = tk.Frame(right, bg=COLORS['bg'])
        self.top_frame.pack(fill=tk.X, padx=12, pady=(0, 6))
        self.top_frame.pack_propagate(False)
        self.top_frame.config(height=60)

        self.top_label = tk.Label(self.top_frame, text='—', font=('Microsoft YaHei', 22, 'bold'),
                                   fg=COLORS['success'], bg=COLORS['bg'])
        self.top_label.pack(expand=True)

        self.top_conf = tk.Label(self.top_frame, text='', font=('Microsoft YaHei', 13),
                                  fg=COLORS['text_dim'], bg=COLORS['bg'])
        self.top_conf.pack()

        ttk.Separator(right, orient='horizontal').pack(fill=tk.X, padx=12, pady=4)

        tk.Label(right, text='Top-5 预测结果', font=('Microsoft YaHei', 12),
                 fg=COLORS['text_dim'], bg=COLORS['panel']).pack(pady=(4, 2))

        self.result_list = tk.Frame(right, bg=COLORS['panel'])
        self.result_list.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 6))

        self.info_frame = tk.Frame(right, bg=COLORS['panel'])
        self.info_frame.pack(fill=tk.X, padx=12, pady=(0, 12))

        self.info_lbl = tk.Label(self.info_frame, text='就绪 — 等待图片输入', font=('Microsoft YaHei', 10),
                                  fg=COLORS['text_dim'], bg=COLORS['panel'])
        self.info_lbl.pack(side=tk.LEFT)

        class_frame = tk.Frame(right, bg=COLORS['bg'])
        class_frame.pack(fill=tk.X, padx=12, pady=(0, 10))

        tk.Label(class_frame, text='支持的类别 (8种)：', font=('Microsoft YaHei', 10, 'bold'),
                 fg=COLORS['text_dim'], bg=COLORS['bg']).pack(anchor=tk.W, padx=8, pady=(6, 2))
        cls_text = tk.Label(class_frame, text='  '.join(CLASSES_ZH), font=('Microsoft YaHei', 11),
                             fg=COLORS['primary'], bg=COLORS['bg'], wraplength=480)
        cls_text.pack(anchor=tk.W, padx=8, pady=(0, 6))

    def _load_model(self):
        try:
            self.model = build_model()
            self.status_lbl.config(text='● 模型就绪', fg=COLORS['success'])
        except Exception as e:
            self.status_lbl.config(text=f'● 模型加载失败: {e}', fg=COLORS['danger'])

    def _select_image(self):
        path = filedialog.askopenfilename(
            title='选择动物图片',
            filetypes=[('图片文件', '*.jpg *.jpeg *.png *.bmp *.webp'), ('所有文件', '*.*')]
        )
        if path:
            self._load_image(path)

    def _load_image(self, path):
        try:
            img = Image.open(path).convert('RGB')
            self.current_image = img
            self._display_image(img)
            self.infer_btn.config(state=tk.NORMAL)
            self._clear_results()
            self.info_lbl.config(text=f'已加载: {os.path.basename(path)} | 大小: {img.size[0]}x{img.size[1]}px')
        except Exception as e:
            messagebox.showerror('错误', f'无法加载图片: {e}')

    def _display_image(self, img):
        canvas_w = self.img_canvas.winfo_width()
        canvas_h = self.img_canvas.winfo_height()
        if canvas_w < 50:
            canvas_w = 490
        if canvas_h < 50:
            canvas_h = 450

        max_w = canvas_w - 20
        max_h = canvas_h - 20
        ratio = min(max_w / img.width, max_h / img.height)
        new_w, new_h = int(img.width * ratio), int(img.height * ratio)
        display = img.resize((new_w, new_h), Image.LANCZOS)

        self.current_tkimg = ImageTk.PhotoImage(display)
        if self.img_canvas_text:
            self.img_canvas.delete(self.img_canvas_text)
            self.img_canvas_text = None
        if self.img_canvas_id:
            self.img_canvas.delete(self.img_canvas_id)
        self.img_canvas_id = self.img_canvas.create_image(
            canvas_w // 2, canvas_h // 2, image=self.current_tkimg, anchor=tk.CENTER
        )

    def _clear(self):
        self.current_image = None
        self.current_tkimg = None
        self.infer_btn.config(state=tk.DISABLED)
        if self.img_canvas_id:
            self.img_canvas.delete(self.img_canvas_id)
            self.img_canvas_id = None
        if not self.img_canvas_text:
            self.img_canvas_text = self.img_canvas.create_text(
                self.img_canvas.winfo_width() // 2 or 260,
                self.img_canvas.winfo_height() // 2 or 220,
                text='点击下方按钮选择图片\n或拖拽图片到此处',
                font=('Microsoft YaHei', 13), fill=COLORS['text_dim'], justify=tk.CENTER
            )
        self._clear_results()
        self.info_lbl.config(text='就绪 — 等待图片输入')

    def _clear_results(self):
        self.top_label.config(text='—', fg=COLORS['success'])
        self.top_conf.config(text='')
        for w in self.result_list.winfo_children():
            w.destroy()
        self.result_data = None

    def _predict(self):
        if self.current_image is None or self.model is None:
            return
        self.infer_btn.config(state=tk.DISABLED, text='  识别中...  ')
        self.status_lbl.config(text='● 推理中...', fg=COLORS['warning'])
        self.root.update()

        try:
            t0 = time.time()
            tensor = TRANSFORM(self.current_image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
            top_probs, top_idx = torch.topk(probs, k=min(5, len(CLASSES_ZH)))
            elapsed = (time.time() - t0) * 1000

            results = []
            for prob, idx in zip(top_probs.tolist(), top_idx.tolist()):
                results.append({
                    'zh': CLASSES_ZH[idx],
                    'en': CLASSES_EN[idx],
                    'prob': prob
                })

            is_unknown = results[0]['prob'] < UNKNOWN_THRESHOLD
            self.result_data = results
            self._show_results(results, elapsed, is_unknown)

        except Exception as e:
            messagebox.showerror('识别错误', f'推理失败: {e}')
        finally:
            self.infer_btn.config(state=tk.NORMAL, text='  开始识别  ')
            self.status_lbl.config(text='● 模型就绪', fg=COLORS['success'])

    def _show_results(self, results, elapsed_ms, is_unknown):
        top = results[0]
        if is_unknown:
            self.top_label.config(text='未知类别', fg=COLORS['warning'])
        else:
            self.top_label.config(text=top['zh'], fg=COLORS['success'])
        self.top_conf.config(text=f'置信度: {top["prob"]*100:.1f}%  |  推理耗时: {elapsed_ms:.1f}ms')

        for w in self.result_list.winfo_children():
            w.destroy()

        for i, r in enumerate(results):
            row = tk.Frame(self.result_list, bg=COLORS['panel'])
            row.pack(fill=tk.X, pady=2)

            rank = f'#{i+1}'
            tk.Label(row, text=rank, font=('Consolas', 12, 'bold'), width=4,
                     fg=COLORS['primary'], bg=COLORS['panel']).pack(side=tk.LEFT, padx=(0, 6))

            name = f'{r["zh"]}'
            tk.Label(row, text=name, font=('Microsoft YaHei', 12, 'bold'), width=8, anchor=tk.W,
                     fg=COLORS['white'], bg=COLORS['panel']).pack(side=tk.LEFT, padx=(0, 8))

            bar_frame = tk.Frame(row, bg=COLORS['bg'], height=22)
            bar_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
            bar_frame.pack_propagate(False)

            bar_color = COLORS['bar_grad'][min(i, len(COLORS['bar_grad'])-1)]
            bar = tk.Frame(bar_frame, bg=bar_color)
            bar.place(relx=0, rely=0, relwidth=r['prob'], relheight=1)

            pct = f'{r["prob"]*100:.1f}%'
            tk.Label(row, text=pct, font=('Consolas', 11), width=8,
                     fg=COLORS['success'] if i == 0 else COLORS['text_dim'],
                     bg=COLORS['panel']).pack(side=tk.RIGHT)

        self.info_lbl.config(
            text=f'推理完成 | 耗时 {elapsed_ms:.1f}ms | {"未知类别 (<60%)" if is_unknown else "识别成功"}'
        )

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    app = RecognitionApp()
    app.run()