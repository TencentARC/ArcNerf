# -*- coding: utf-8 -*-

import torch


def run_eval(
    loader,
    get_model_feed_in,
    model,
    logger,
    eval_metric,
    metric_summary,
    device,
    render_progress_img,
    max_samples_eval=None,
    show_progress=True
):
    """Core eval function for evaluation
       We turn it to eval() mode at beginning but don't turn it to train() mode at the end,
       you need to do it outside this func if needed
    """
    model.eval()
    count = 0
    files = None
    with torch.no_grad():
        for step, inputs in enumerate(loader):
            if show_progress:
                logger.add_log('Progress : {}/{}'.format(step, len(loader) - 1))
            feed_in, batch_size = get_model_feed_in(inputs, device)
            output = model(feed_in)

            if max_samples_eval is None or step < max_samples_eval:
                if files is None:
                    files = []
                files.append(render_progress_img(inputs, output))

            count += batch_size
            metrics = eval_metric(inputs, output)
            metric_summary(metrics, batch_size)

        if count == 0:
            logger.add_log('Not batch was sent to eval...')
            return

        # get average
        metric_summary.cal_average()

    return metric_summary.get_metric_info(), files
