# -*- coding: utf-8 -*-

import time

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
    total_forward_time = 0.0
    with torch.no_grad():
        for step, inputs in enumerate(loader):
            if show_progress:
                logger.add_log('Progress : {}/{}'.format(step, len(loader) - 1))
            feed_in, batch_size = get_model_feed_in(inputs, device)
            time0 = time.time()
            output = model(feed_in, inference_only=True)
            total_forward_time += (time.time() - time0)
            h, w = int(inputs['H'][0]), int(inputs['W'][0])

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

        avg_time = total_forward_time / float(count)
        logger.add_log('   Eval sample (H,W)=({},{}) avg forward time {:.2f}s'.format(h, w, avg_time))

        # get average
        metric_summary.cal_average()

    return metric_summary.get_metric_info(), files
