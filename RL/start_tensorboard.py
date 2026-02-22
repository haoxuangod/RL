
if __name__ == "__main__":
    from tensorboard import program
    tb = program.TensorBoard()
    tensorboard_dir = './tensorboard_record_cartPole-v1/'
    tb.configure(argv=[None, '--logdir', tensorboard_dir, '--port', '6006'])
    url = tb.launch()
    print(f"TensorBoard listening on {url}")
    while 1:
        pass