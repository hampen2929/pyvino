import logging
from logging import getLogger, StreamHandler, Formatter


def get_logger(name):

    # --------------------------------
    # 1.loggerの設定
    # --------------------------------
    # loggerオブジェクトの宣言
    logger = getLogger(name)

    # loggerのログレベル設定(ハンドラに渡すエラーメッセージのレベル)
    logger.setLevel(logging.DEBUG)

    # --------------------------------
    # 2.handlerの設定
    # --------------------------------
    # handlerの生成
    stream_handler = StreamHandler()

    # handlerのログレベル設定(ハンドラが出力するエラーメッセージのレベル)
    stream_handler.setLevel(logging.DEBUG)

    # ログ出力フォーマット設定
    handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(handler_format)

    # --------------------------------
    # 3.loggerにhandlerをセット
    # --------------------------------
    logger.addHandler(stream_handler)
    return logger


