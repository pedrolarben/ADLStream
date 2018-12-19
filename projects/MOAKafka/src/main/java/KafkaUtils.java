import kafka.admin.AdminUtils;
import kafka.admin.RackAwareMode;
import kafka.utils.ZKStringSerializer$;
import kafka.utils.ZkUtils;
import org.I0Itec.zkclient.ZkClient;
import org.I0Itec.zkclient.ZkConnection;

import java.util.Properties;

public class KafkaUtils {

    public static void createTopic(String topicName, int numPartitions, int numReplications){
        System.out.println("Creating new topic with name: " + topicName);
        ZkClient zkClient = null;
        ZkUtils zkUtils = null;
        try {
            // If multiple zookeeper then -> String zookeeperHosts = "192.168.1.10:2181,192.168.1.11:2181";
            String zookeeperHosts = "localhost:2181";
            int sessionTimeOutInMs = 15 * 1000; // 15 secs
            int connectionTimeOutInMs = 10 * 1000; // 10 secs

            zkClient = new ZkClient(zookeeperHosts, sessionTimeOutInMs,
                    connectionTimeOutInMs, ZKStringSerializer$.MODULE$);
            zkUtils = new ZkUtils(zkClient, new ZkConnection(zookeeperHosts), false);

            Properties topicConfiguration = new Properties();

            AdminUtils.createTopic(zkUtils, topicName, numPartitions, numReplications,
                    topicConfiguration, RackAwareMode.Enforced$.MODULE$);

        } catch (Exception ex) {
            ex.printStackTrace();
        } finally {
            if (zkClient != null) {
                zkClient.close();
            }
        }
    }

    public static boolean existsTopic(String topicName){
        ZkClient zkClient = null;
        ZkUtils zkUtils = null;
        boolean exists = false;
        try {
            // If multiple zookeeper then -> String zookeeperHosts = "192.168.1.10:2181,192.168.1.11:2181";
            String zookeeperHosts = "localhost:2181";
            int sessionTimeOutInMs = 15 * 1000; // 15 secs
            int connectionTimeOutInMs = 10 * 1000; // 10 secs

            zkClient = new ZkClient(zookeeperHosts, sessionTimeOutInMs,
                    connectionTimeOutInMs, ZKStringSerializer$.MODULE$);
            zkUtils = new ZkUtils(zkClient, new ZkConnection(zookeeperHosts), false);

            exists = AdminUtils.topicExists(zkUtils, topicName);

        } catch (Exception ex) {
            ex.printStackTrace();
        } finally {
            if (zkClient != null) {
                zkClient.close();
            }
        }
        return exists;
    }
}
