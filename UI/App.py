# import customtkinter as ctk
# from login import LoginFrame
# from Dashboard import DashboardFrame

# class App(ctk.CTk):
#     def __init__(self):
#         super().__init__()
#         self.geometry("1200x800")
#         self.title("DefectScan Pro")

#         self.current_frame = None
#         self.show_login()

#     def clear_frame(self):
#         if self.current_frame:
#             self.current_frame.destroy()

#     def show_login(self):
#         self.clear_frame()
#         self.current_frame = LoginFrame(self, self.show_dashboard)
#         self.current_frame.pack(expand=True, fill="both")

#     def show_dashboard(self):
#         self.clear_frame()
#         self.current_frame = DashboardFrame(self, self.show_login)
#         self.current_frame.pack(expand=True, fill="both")


# if __name__ == "__main__":
#     app = App()
#     app.mainloop()
