import taichi as ti
import taichi.math as tm
import time


@ti.data_oriented
class BaseShader:

    def __init__(self,
                 title: str,
                 res: tuple[int, int] | None = None,
                 gamma: float = 2.2
                 ):
        self.title = title
        self.res = res if res is not None else (1000, 563)
        self.resf = tm.vec2(*self.res)
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=self.res)
        self.gamma = gamma

    @ti.kernel
    def init(self):
        pass

    @ti.kernel
    def calculate(self, t: ti.f32):
        pass

    @ti.func
    def main_image(self, uv, t):
        col = tm.vec3(0.)
        col.rg = uv + 0.5
        return col

    @ti.kernel
    def render(self, t: ti.f32):
        for fragCoord in ti.grouped(self.pixels):
            uv = (fragCoord - 0.5 * self.resf) / self.resf.y
            col = self.main_image(uv, t)
            if self.gamma > 0.0:
                col = tm.clamp(col ** (1 / self.gamma), 0., 1.)
            self.pixels[fragCoord] = col

    def main_loop(self):
        gui = ti.GUI(self.title, res=self.res, fast_gui=True)
        start = time.time()

        self.init()
        while gui.running:  # основной цикл
            if gui.get_event(ti.GUI.PRESS):  # для закрытия приложения по нажатию на Esc
                if gui.event.key == ti.GUI.ESCAPE:
                    break

            t = time.time() - start  # пересчет времени, прошедшего с первого кадра
            self.calculate(t)
            self.render(t)  # расчет цветов пикселей
            gui.set_image(self.pixels)  # перенос пикселей из поля pixels в буфер кадра
            gui.show()

        gui.close()


class TwoPassShader(BaseShader):

    def __init__(self,
                 title: str,
                 res: tuple[int, int] | None = None,
                 gamma: float = 2.2
                 ):
        super().__init__(title, res=res, gamma=gamma)
        self.buffer = ti.Vector.field(3, dtype=ti.f32, shape=res)

    @ti.kernel
    def render_pass1(self, t: ti.f32):
        for fragCoord in ti.grouped(self.buffer):
            uv = (fragCoord - 0.5 * self.resf) / self.resf.y
            col = self.main_image(uv, t)
            if self.gamma > 0.0:
                col = tm.clamp(col ** (1 / self.gamma), 0., 1.)
            self.buffer[fragCoord] = col

    @ti.kernel
    def render_pass2(self, t: ti.f32):
        for fragCoord in ti.grouped(self.pixels):
            col = self.buffer[fragCoord // 16 * 16]
            self.pixels[fragCoord] = col

    def render(self, t):
        self.render_pass1(t)
        self.render_pass2(t)

    def main_loop(self):
        gui = ti.GUI(self.title, res=self.res, fast_gui=True)
        start = time.time()
        show_buffer = False

        while gui.running:  # основной цикл
            if gui.get_event(ti.GUI.PRESS):  # для закрытия приложения по нажатию на Esc
                if gui.event.key == ti.GUI.ESCAPE:
                    break
                elif gui.event.key == ti.GUI.RETURN:
                    show_buffer = not show_buffer

            t = time.time() - start  # пересчет времени, прошедшего с первого кадра
            self.render(t)  # расчет цветов пикселей
            if show_buffer:
                gui.set_image(self.buffer)  # перенос пикселей из поля buffer в буфер кадра
            else:
                gui.set_image(self.pixels)  # перенос пикселей из поля pixels в буфер кадра
            gui.show()

        gui.close()



if __name__ == "__main__":

    ti.init(arch=ti.opengl)

    shader = BaseShader("Base shader")

    # shader = TwoPassShader("Two pass shader | 16x16 blocks")

    shader.main_loop()
