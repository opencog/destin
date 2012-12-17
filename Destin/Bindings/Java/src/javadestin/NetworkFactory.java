/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package javadestin;

import callbacks.BeliefGraphCallback;

/**
 *
 * @author teds
 */
public class NetworkFactory implements INetworkFactory {

    @Override
    public INetwork create() {
            INetwork n = new NetworkAlt(SupportedImageWidths.W512, 8, new long[]{20, 16, 14, 12, 10, 8, 6, 2}, true);
            n.setIterationFinishedCallback(new BeliefGraphCallback());
            return n;
    }
}
